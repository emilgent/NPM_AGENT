/**
 * src/tools.js
 * ----------------------------------------------------------------------------
 * System I/O and External Integration Layer.
 *
 * This module houses the six native tool implementations that the autonomous
 * agent is allowed to invoke through the OpenAI-compatible function-calling
 * interface:
 *
 *   1. listFiles()          - recursive project tree listing
 *   2. readFile()           - safe text file reader (500 KB ceiling)
 *   3. writeFile()          - text file writer with recursive mkdir
 *   4. executeCommand()     - guarded shell execution with whitelist + approval
 *   5. webSearch()          - SerpAPI powered web search
 *   6. fetchWebpage()       - HTTP GET + HTML to readable-text cleaner
 *
 * Cross-cutting concerns implemented here:
 *   - A binary-file safeguard that refuses to touch non-text files.
 *   - An "intelligent" safety interceptor for shell commands that distinguishes
 *     read-only / diagnostic commands from state-mutating commands and asks the
 *     human operator to confirm the latter.
 *   - A hard file-size validation wall for reads.
 *   - A lightweight HTML cleaning utility for documentation scraping.
 *
 * The module intentionally throws descriptive Error objects (whose `.message`
 * carries the exact text mandated by the specification). The agent scheduler
 * captures those messages and feeds them straight back to the LLM as the
 * content of a `tool` role message, allowing the model to self-correct.
 * ----------------------------------------------------------------------------
 */

import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

import chalk from 'chalk';
import inquirer from 'inquirer';
import fetch from 'node-fetch';

// Promisified variant of child_process.exec so we can `await` shell commands.
const execAsync = promisify(exec);

// ---------------------------------------------------------------------------
// Configuration constants
// ---------------------------------------------------------------------------

/**
 * The maximum size (in bytes) of a file that `readFile` is willing to load into
 * the model context. 500 KB keeps the context window from being blown out by a
 * single oversized artefact.
 */
const MAX_READ_FILE_BYTES = 500 * 1024; // 500 KB

/**
 * Hard ceiling for any shell command executed through `executeCommand`.
 * Long build / test pipelines may run for a while, hence the generous one hour.
 */
const COMMAND_TIMEOUT_MS = 60 * 60 * 1000; // 3,600,000 ms (1 hour)

/**
 * Directories that the recursive file walker must never descend into. These are
 * either huge (node_modules), irrelevant to source reasoning (.git) or internal
 * agent bookkeeping (.agent_sessions). Skipping them preserves context size.
 */
const IGNORED_DIRECTORIES = new Set(['node_modules', '.git', '.agent_sessions']);

/**
 * Explicitly recognised binary file extensions. Encountering any of these in a
 * read or write request triggers the binary safeguard immediately.
 */
const KNOWN_BINARY_EXTENSIONS = new Set([
  '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.tiff',
  '.zip', '.tar', '.gz', '.tgz', '.rar', '.7z', '.bz2',
  '.exe', '.dll', '.so', '.dylib', '.bin', '.o', '.a',
  '.node', '.wasm',
  '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
  '.mp3', '.mp4', '.wav', '.avi', '.mov', '.mkv', '.flac',
  '.ttf', '.otf', '.woff', '.woff2', '.eot',
  '.class', '.jar', '.pyc', '.pdb',
]);

/**
 * Allowlist of text-based formats the agent is permitted to read and write.
 * The specification mandates that ONLY text formats are processed; anything
 * outside of this list is rejected by the binary safeguard.
 */
const ALLOWED_TEXT_EXTENSIONS = new Set([
  '.js', '.mjs', '.cjs', '.jsx',
  '.ts', '.tsx',
  '.json', '.jsonc',
  '.md', '.markdown',
  '.html', '.htm',
  '.css', '.scss', '.sass', '.less',
  '.yml', '.yaml',
  '.txt', '.xml', '.svg', '.csv',
  '.env', '.sh', '.toml', '.ini', '.cfg',
]);

/**
 * The exact, specification-mandated error message used whenever a binary file
 * manipulation is attempted.
 */
const BINARY_FILE_ERROR = 'Error: Binary file manipulation is strictly prohibited.';

/**
 * The exact, specification-mandated error message used when a file exceeds the
 * processing size limit.
 */
const FILE_SIZE_ERROR = 'Error: File size exceeds the maximum processing limit of 500KB.';

/**
 * The exact observation that is fed back to the LLM when the human operator
 * rejects a state-mutating shell command.
 */
const COMMAND_REJECTED_OBSERVATION =
  'Error: Command rejected and blocked by the user due to safety policies.';

// ---------------------------------------------------------------------------
// Safety helpers
// ---------------------------------------------------------------------------

/**
 * Inspect a file path and enforce the text-only safety boundary.
 *
 * The check is purely extension based: any extension explicitly known to be
 * binary, or any extension that is not present in the text allowlist, is
 * rejected. Dot-prefixed config files without a real extension (e.g.
 * ".gitignore", ".npmrc", ".eslintrc") are treated as text because they are
 * conventionally plain text.
 *
 * @param {string} filePath - The target file path to validate.
 * @throws {Error} If the file is considered binary / non-text.
 */
function assertTextFile(filePath) {
  const baseName = path.basename(filePath);
  const extension = path.extname(baseName).toLowerCase();

  // Files such as ".gitignore" or "Dockerfile" report an empty extension via
  // path.extname; treat dot-files and well-known extensionless config files as
  // text so the agent can still operate on them.
  if (extension === '') {
    const dotFile = baseName.startsWith('.');
    const knownExtensionlessText = new Set([
      'Dockerfile', 'Makefile', 'LICENSE', 'README', 'Procfile', '.gitignore',
      '.npmrc', '.npmignore', '.nvmrc', '.editorconfig', '.prettierrc',
      '.eslintrc', '.babelrc', '.env',
    ]);
    if (dotFile || knownExtensionlessText.has(baseName)) {
      return;
    }
    // Unknown extensionless binaries (compiled executables etc.) are blocked.
    throw new Error(BINARY_FILE_ERROR);
  }

  if (KNOWN_BINARY_EXTENSIONS.has(extension)) {
    throw new Error(BINARY_FILE_ERROR);
  }

  if (!ALLOWED_TEXT_EXTENSIONS.has(extension)) {
    // Anything not explicitly recognised as text is rejected. This satisfies
    // the "allow text-based formats only" requirement.
    throw new Error(BINARY_FILE_ERROR);
  }
}

/**
 * Regular expression describing read-only / diagnostic commands that may run
 * without interactive confirmation. Matching is anchored at the start of the
 * (trimmed) command string so that, e.g., `npm test -- --watch` still matches
 * while `npm test && rm -rf /` does NOT (the chained mutation is caught below).
 */
const SAFE_COMMAND_PATTERN = new RegExp(
  [
    '^npm (test|run build|run lint|run test|ci-info|view|ls|outdated|audit)\\b',
    '^node\\s+\\S+',
    '^npx\\s+(tsc|eslint|prettier --check|vitest run|jest)\\b',
    '^git (status|diff|log|branch|show|rev-parse|remote -v)\\b',
    '^(ls|pwd|cat|echo|head|tail|wc|grep|find|tree|which|whoami|date|env)\\b',
    '^tsc\\b',
  ].join('|'),
);

/**
 * Regular expression describing commands that mutate filesystem / system /
 * package state. Matching any of these forces an interactive approval prompt.
 */
const MUTATING_COMMAND_PATTERN = new RegExp(
  [
    '\\bnpm (install|i|uninstall|remove|rm|update|upgrade|publish|link|prune|dedupe)\\b',
    '\\b(yarn|pnpm)\\s+(add|remove|install|up|upgrade)\\b',
    '\\brm\\b',
    '\\brmdir\\b',
    '\\bmkdir\\b',
    '\\bmv\\b',
    '\\bcp\\b',
    '\\btouch\\b',
    '\\bdel\\b',
    '\\bchmod\\b',
    '\\bchown\\b',
    '\\bgit (add|commit|push|reset|checkout|merge|rebase|clean|stash)\\b',
    '\\b(sudo|apt|apt-get|brew|curl|wget|kill|pkill|systemctl|service)\\b',
    '>{1,2}', // output redirection writes to disk
  ].join('|'),
);

/**
 * Decide whether a shell command requires interactive human approval.
 *
 * A command is considered "safe" (auto-runnable) only when it matches the
 * read-only whitelist AND does NOT also match any mutating pattern (to defend
 * against chained commands such as `git status && rm -rf build`).
 *
 * @param {string} command - The shell command to classify.
 * @returns {boolean} True if approval is required, false if safe to auto-run.
 */
function commandRequiresApproval(command) {
  const trimmed = command.trim();

  // If it contains a mutating token anywhere, always require approval.
  if (MUTATING_COMMAND_PATTERN.test(trimmed)) {
    return true;
  }

  // If it clearly matches a known read-only command, allow it through.
  if (SAFE_COMMAND_PATTERN.test(trimmed)) {
    return false;
  }

  // Unknown commands default to requiring approval (fail closed).
  return true;
}

// ---------------------------------------------------------------------------
// Tool 1: listFiles
// ---------------------------------------------------------------------------

/**
 * Recursively walk the current working directory and produce a structured,
 * indented text listing of files and subdirectories. The directories listed in
 * IGNORED_DIRECTORIES are skipped entirely.
 *
 * @returns {Promise<string>} A human/LLM readable directory tree.
 */
export async function listFiles() {
  const root = process.cwd();
  const lines = [];

  /**
   * Inner recursive walker.
   * @param {string} dir - Absolute directory to scan.
   * @param {string} prefix - Indentation prefix for the current depth.
   */
  async function walk(dir, prefix) {
    let entries;
    try {
      entries = await fsp.readdir(dir, { withFileTypes: true });
    } catch (err) {
      lines.push(`${prefix}[unreadable directory: ${err.message}]`);
      return;
    }

    // Stable, deterministic ordering: directories first, then files, alpha sort.
    entries.sort((a, b) => {
      if (a.isDirectory() && !b.isDirectory()) return -1;
      if (!a.isDirectory() && b.isDirectory()) return 1;
      return a.name.localeCompare(b.name);
    });

    for (const entry of entries) {
      if (entry.isDirectory()) {
        if (IGNORED_DIRECTORIES.has(entry.name)) {
          lines.push(`${prefix}${entry.name}/ (skipped)`);
          continue;
        }
        lines.push(`${prefix}${entry.name}/`);
        await walk(path.join(dir, entry.name), `${prefix}  `);
      } else if (entry.isFile()) {
        lines.push(`${prefix}${entry.name}`);
      } else {
        // Symlinks, sockets, etc. are reported but not followed.
        lines.push(`${prefix}${entry.name} (special)`);
      }
    }
  }

  lines.push(`${path.basename(root)}/ (cwd: ${root})`);
  await walk(root, '  ');

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Tool 2: readFile
// ---------------------------------------------------------------------------

/**
 * Read the textual contents of a file relative to (or absolute within) the
 * current working directory.
 *
 * Enforces two safety walls:
 *   - The binary-file safeguard (assertTextFile).
 *   - The 500 KB maximum file size limit.
 *
 * @param {string} filePath - Path of the file to read.
 * @returns {Promise<string>} The file contents as UTF-8 text.
 * @throws {Error} On binary files, oversized files or read failures.
 */
export async function readFile(filePath) {
  if (typeof filePath !== 'string' || filePath.trim() === '') {
    throw new Error('Error: readFile requires a non-empty "filePath" argument.');
  }

  // Enforce the text-only boundary before touching the filesystem.
  assertTextFile(filePath);

  const resolved = path.resolve(process.cwd(), filePath);

  let stats;
  try {
    stats = await fsp.stat(resolved);
  } catch (err) {
    throw new Error(`Error: Unable to access file "${filePath}": ${err.message}`);
  }

  if (!stats.isFile()) {
    throw new Error(`Error: "${filePath}" is not a regular file.`);
  }

  // Hard size validation wall.
  if (stats.size > MAX_READ_FILE_BYTES) {
    throw new Error(FILE_SIZE_ERROR);
  }

  const content = await fsp.readFile(resolved, 'utf8');
  return content;
}

// ---------------------------------------------------------------------------
// Tool 3: writeFile
// ---------------------------------------------------------------------------

/**
 * Write raw string content to a file, creating any missing parent directories
 * recursively. Subject to the same binary-file safeguard as readFile.
 *
 * @param {string} filePath - Destination path.
 * @param {string} content - Raw text content to write.
 * @returns {Promise<string>} A confirmation summary string.
 * @throws {Error} On binary files or write failures.
 */
export async function writeFile(filePath, content) {
  if (typeof filePath !== 'string' || filePath.trim() === '') {
    throw new Error('Error: writeFile requires a non-empty "filePath" argument.');
  }
  if (typeof content !== 'string') {
    // Coerce non-string content defensively; the LLM occasionally sends objects.
    content = content === undefined || content === null ? '' : String(content);
  }

  // Enforce the text-only boundary before touching the filesystem.
  assertTextFile(filePath);

  const resolved = path.resolve(process.cwd(), filePath);
  const dir = path.dirname(resolved);

  // Recursively create parent directories if they do not exist.
  await fsp.mkdir(dir, { recursive: true });

  await fsp.writeFile(resolved, content, 'utf8');

  const byteLength = Buffer.byteLength(content, 'utf8');
  return `Successfully wrote ${byteLength} bytes to "${filePath}".`;
}

// ---------------------------------------------------------------------------
// Tool 4: executeCommand
// ---------------------------------------------------------------------------

/**
 * Execute an external shell command with a one-hour timeout, applying the
 * intelligent safety interceptor.
 *
 * Behaviour:
 *   - Read-only / diagnostic commands run immediately.
 *   - State-mutating commands are rendered in magenta and gated behind an
 *     interactive Inquirer Y/N confirmation. A "No" answer aborts and returns
 *     the canonical rejection observation to the model.
 *   - Non-zero exit codes do not throw; instead the captured stdout/stderr is
 *     wrapped and returned so the model can debug autonomously.
 *
 * @param {string} command - The shell command to execute.
 * @returns {Promise<string>} The combined stdout/stderr (or error) observation.
 */
export async function executeCommand(command) {
  if (typeof command !== 'string' || command.trim() === '') {
    throw new Error('Error: executeCommand requires a non-empty "command" argument.');
  }

  // ----- Intelligent safety check -----------------------------------------
  if (commandRequiresApproval(command)) {
    // High-visibility approval banner (bright magenta, bold) for the operator.
    console.log(
      chalk.magentaBright.bold('\n[USER APPROVAL] A state-mutating command requires your confirmation:'),
    );
    console.log(chalk.magenta(`  > ${command}\n`));

    const { approved } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'approved',
        message: chalk.magentaBright('Allow this command to run?'),
        default: false,
      },
    ]);

    if (!approved) {
      // Feed the exact, specification-mandated observation back to the LLM.
      console.log(chalk.red('  Command blocked by user.\n'));
      return COMMAND_REJECTED_OBSERVATION;
    }
  }

  // ----- Execution ---------------------------------------------------------
  try {
    const { stdout, stderr } = await execAsync(command, {
      timeout: COMMAND_TIMEOUT_MS,
      cwd: process.cwd(),
      maxBuffer: 10 * 1024 * 1024, // 10 MB output buffer
      shell: true,
    });

    const out = (stdout || '').toString();
    const err = (stderr || '').toString();

    let observation = `$ ${command}\n--- exit code: 0 ---\n`;
    if (out.trim()) observation += `STDOUT:\n${out}\n`;
    if (err.trim()) observation += `STDERR:\n${err}\n`;
    if (!out.trim() && !err.trim()) observation += '(no output)\n';

    return observation;
  } catch (err) {
    // execAsync rejects on non-zero exit or timeout. Capture everything and
    // return it as the observation so the model can self-correct.
    const code = err.code !== undefined ? err.code : 'unknown';
    const killed = err.killed ? ' (process killed — possible timeout)' : '';
    const out = (err.stdout || '').toString();
    const stderrBuf = (err.stderr || '').toString();

    let observation = `$ ${command}\n--- exit code: ${code}${killed} ---\n`;
    if (out.trim()) observation += `STDOUT:\n${out}\n`;
    if (stderrBuf.trim()) observation += `STDERR:\n${stderrBuf}\n`;
    if (!out.trim() && !stderrBuf.trim()) {
      observation += `ERROR:\n${err.message}\n`;
    }

    return observation;
  }
}

// ---------------------------------------------------------------------------
// Tool 5: webSearch
// ---------------------------------------------------------------------------

/**
 * Perform a web search via SerpAPI's Google engine.
 *
 * The API key is read from the SERPAPI_KEY environment variable. A configurable
 * fallback constant is provided so the tool can be wired to a different default
 * during local experimentation without editing the call sites.
 *
 * @param {string} query - The natural-language search query.
 * @returns {Promise<string>} A simplified text summary (Title / Link / Snippet).
 * @throws {Error} When no API key is configured or the request fails.
 */
const SERPAPI_FALLBACK_KEY = ''; // Optional hard-coded fallback (intentionally empty).

export async function webSearch(query) {
  if (typeof query !== 'string' || query.trim() === '') {
    throw new Error('Error: webSearch requires a non-empty "query" argument.');
  }

  const apiKey = process.env.SERPAPI_KEY || SERPAPI_FALLBACK_KEY;
  if (!apiKey) {
    throw new Error(
      'Error: No SerpAPI key configured. Set the SERPAPI_KEY environment variable to enable web search.',
    );
  }

  const endpoint = new URL('https://serpapi.com/search.json');
  endpoint.searchParams.set('engine', 'google');
  endpoint.searchParams.set('q', query);
  endpoint.searchParams.set('api_key', apiKey);
  endpoint.searchParams.set('num', '8');

  const response = await fetch(endpoint.toString(), { method: 'GET' });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(
      `Error: SerpAPI request failed with status ${response.status} ${response.statusText}. ${body.slice(0, 300)}`,
    );
  }

  const payload = await response.json();

  // Aggregate the most useful organic results into a compact summary.
  const organic = Array.isArray(payload.organic_results) ? payload.organic_results : [];
  if (organic.length === 0) {
    return `No organic results found for query: "${query}".`;
  }

  const summaries = organic.slice(0, 8).map((result, index) => {
    const title = result.title || '(no title)';
    const link = result.link || '(no link)';
    const snippet = result.snippet || result.snippet_highlighted_words?.join(' ') || '(no snippet)';
    return `${index + 1}. ${title}\n   Link: ${link}\n   Snippet: ${snippet}`;
  });

  return `Web search results for "${query}":\n\n${summaries.join('\n\n')}`;
}

// ---------------------------------------------------------------------------
// Tool 6: fetchWebpage
// ---------------------------------------------------------------------------

/**
 * Lightweight, dependency-free HTML to readable-text cleaner.
 *
 * Strips <script>, <style>, <noscript>, <svg>, <head>, navigational chrome
 * (nav / header / footer / aside), all remaining tags, decodes a handful of
 * common HTML entities and collapses excess whitespace.
 *
 * @param {string} html - Raw HTML markup.
 * @returns {string} Cleaned, readable plain text.
 */
function cleanHtmlToText(html) {
  let text = html;

  // Remove entire blocks whose inner content is never useful to the agent.
  const blockTags = ['script', 'style', 'noscript', 'svg', 'head', 'nav', 'header', 'footer', 'aside', 'form'];
  for (const tag of blockTags) {
    const blockRegex = new RegExp(`<${tag}\\b[^>]*>[\\s\\S]*?<\\/${tag}>`, 'gi');
    text = text.replace(blockRegex, ' ');
  }

  // Convert common block-level closing tags into newlines to preserve layout.
  text = text.replace(/<\/(p|div|li|h[1-6]|tr|section|article|br)\s*>/gi, '\n');
  text = text.replace(/<br\s*\/?>/gi, '\n');

  // Strip every remaining HTML tag.
  text = text.replace(/<[^>]+>/g, ' ');

  // Decode a small set of frequently encountered HTML entities.
  const entities = {
    '&nbsp;': ' ',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&apos;': "'",
    '&mdash;': '-',
    '&ndash;': '-',
    '&hellip;': '...',
  };
  for (const [entity, replacement] of Object.entries(entities)) {
    text = text.replace(new RegExp(entity, 'g'), replacement);
  }
  // Numeric entities (decimal).
  text = text.replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)));

  // Collapse runs of whitespace while keeping paragraph breaks readable.
  text = text
    .split('\n')
    .map((line) => line.replace(/[ \t\f\v]+/g, ' ').trim())
    .filter((line) => line.length > 0)
    .join('\n');

  return text;
}

/**
 * Fetch a webpage and return cleaned, readable text (useful for scraping docs).
 *
 * @param {string} url - The absolute URL to fetch (http/https).
 * @returns {Promise<string>} Cleaned page text (truncated to a sane length).
 * @throws {Error} On invalid URLs or failed requests.
 */
export async function fetchWebpage(url) {
  if (typeof url !== 'string' || url.trim() === '') {
    throw new Error('Error: fetchWebpage requires a non-empty "url" argument.');
  }

  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Error: "${url}" is not a valid absolute URL.`);
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Error: fetchWebpage only supports http and https URLs.');
  }

  const response = await fetch(parsed.toString(), {
    method: 'GET',
    headers: {
      'User-Agent':
        'Mozilla/5.0 (compatible; npm-agent/1.0; +https://github.com/emilgent/NPM_AGENT)',
      Accept: 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8',
    },
    redirect: 'follow',
  });

  if (!response.ok) {
    throw new Error(
      `Error: Failed to fetch "${url}" — HTTP ${response.status} ${response.statusText}.`,
    );
  }

  const contentType = response.headers.get('content-type') || '';
  const raw = await response.text();

  // Plain text responses do not need HTML cleaning.
  const cleaned = contentType.includes('text/html') ? cleanHtmlToText(raw) : raw;

  // Guard the context window: cap the returned text at ~20k characters.
  const MAX_CHARS = 20000;
  if (cleaned.length > MAX_CHARS) {
    return `${cleaned.slice(0, MAX_CHARS)}\n\n[...truncated ${cleaned.length - MAX_CHARS} characters...]`;
  }
  return cleaned;
}

// ---------------------------------------------------------------------------
// Tool registry + dispatcher
// ---------------------------------------------------------------------------

/**
 * Map of tool name -> implementation. The agent scheduler uses this registry to
 * dispatch `tool_calls` returned by the model.
 */
export const toolImplementations = {
  listFiles,
  readFile,
  writeFile,
  executeCommand,
  webSearch,
  fetchWebpage,
};

/**
 * OpenAI-compatible JSON schema descriptors for every tool. This array is sent
 * verbatim as the `tools` field of each chat-completion request so the model
 * knows which functions it may call and with what arguments.
 */
export const toolSchemas = [
  {
    type: 'function',
    function: {
      name: 'listFiles',
      description:
        'Recursively list all files and subdirectories in the current working directory. ' +
        'Automatically skips node_modules, .git and .agent_sessions. Takes no arguments.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'readFile',
      description:
        'Read the UTF-8 text contents of a single file. Rejects binary files and files larger than 500 KB.',
      parameters: {
        type: 'object',
        properties: {
          filePath: {
            type: 'string',
            description: 'Path to the file, relative to the current working directory or absolute.',
          },
        },
        required: ['filePath'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'writeFile',
      description:
        'Write raw text content to a file, creating any missing parent directories. Rejects binary files.',
      parameters: {
        type: 'object',
        properties: {
          filePath: {
            type: 'string',
            description: 'Destination path, relative to the current working directory or absolute.',
          },
          content: {
            type: 'string',
            description: 'The full raw text content to write to the file.',
          },
        },
        required: ['filePath', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'executeCommand',
      description:
        'Execute a shell command in the current working directory (1 hour timeout). ' +
        'Read-only/diagnostic commands run automatically; state-mutating commands require explicit user approval.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'The exact shell command to execute.',
          },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'webSearch',
      description:
        'Search the web via SerpAPI and return a list of Title / Link / Snippet summaries for the query.',
      parameters: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'The search query string.',
          },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'fetchWebpage',
      description:
        'Fetch an http/https URL and return cleaned, readable plain text (scripts, styles and navigation removed).',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The absolute http/https URL to fetch.',
          },
        },
        required: ['url'],
      },
    },
  },
];

/**
 * Dispatch a single tool call by name with a parsed arguments object.
 *
 * @param {string} name - The tool/function name requested by the model.
 * @param {object} args - The parsed arguments object.
 * @returns {Promise<string>} The string observation produced by the tool.
 */
export async function dispatchTool(name, args) {
  const impl = toolImplementations[name];
  if (!impl) {
    return `Error: Unknown tool "${name}". Available tools: ${Object.keys(toolImplementations).join(', ')}.`;
  }

  switch (name) {
    case 'listFiles':
      return impl();
    case 'readFile':
      return impl(args.filePath);
    case 'writeFile':
      return impl(args.filePath, args.content);
    case 'executeCommand':
      return impl(args.command);
    case 'webSearch':
      return impl(args.query);
    case 'fetchWebpage':
      return impl(args.url);
    default:
      return `Error: No dispatch mapping defined for tool "${name}".`;
  }
}

// Re-export the safety constants so other modules / tests can reference the
// exact mandated strings without duplicating them.
export const SAFETY_MESSAGES = {
  BINARY_FILE_ERROR,
  FILE_SIZE_ERROR,
  COMMAND_REJECTED_OBSERVATION,
};
