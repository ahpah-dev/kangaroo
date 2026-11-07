'use strict';
// Spawns and supervises the Python engine. Kept free of Electron imports so it
// can be unit-tested with plain `node --test`.
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const COMMANDS = new Set(['train', 'generate', 'export', 'info']);
const REPO_ROOT = path.resolve(__dirname, '..');

function resolvePython(preferred) {
  if (preferred && String(preferred).trim()) return String(preferred).trim();
  if (process.env.KANGAROO_PYTHON) return process.env.KANGAROO_PYTHON;
  const venv = process.platform === 'win32'
    ? path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    : path.join(REPO_ROOT, '.venv', 'bin', 'python');
  if (fs.existsSync(venv)) return venv;
  return process.platform === 'win32' ? 'python' : 'python3';
}

/** Split a byte stream into complete lines, tolerating chunk boundaries. */
function lineSplitter(onLine) {
  let buf = '';
  return (chunk) => {
    buf += chunk.toString('utf8');
    let i;
    while ((i = buf.search(/\r\n|\n|\r/)) !== -1) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + (buf.startsWith('\r\n', i) ? 2 : 1));
      if (line.trim()) onLine(line);
    }
  };
}

function killTree(child) {
  if (!child || child.exitCode !== null) return;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
  } else {
    child.kill('SIGKILL');
  }
}

class Engine {
  constructor(onEvent) {
    this.onEvent = onEvent;
    this.child = null;
    this.kind = null;
    this.finishing = false;
    this.exited = Promise.resolve();
    this.stopFile = null;
    this.killTimer = null;
  }

  get running() { return this.child !== null; }

  /** Start a job. Resolves when the process exits. */
  start(command, config, pythonPath) {
    if (!COMMANDS.has(command)) throw new Error(`Unknown command: ${command}`);
    if (this.child) throw new Error('A job is already running.');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kangaroo-'));
    const cfgFile = path.join(dir, 'config.json');
    this.stopFile = path.join(dir, 'STOP');
    const cfg = { ...(config || {}) };
    if (command === 'train') cfg.stop_file = this.stopFile;
    fs.writeFileSync(cfgFile, JSON.stringify(cfg), 'utf8');

    const args = ['-m', 'kangaroo', command];
    if (command !== 'info') args.push('--config', cfgFile);
    const env = {
      ...process.env,
      PYTHONPATH: [path.join(REPO_ROOT, 'src'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
      PYTHONUNBUFFERED: '1',
      PYTHONUTF8: '1',
      PYTHONIOENCODING: 'utf-8',
      HF_HUB_DISABLE_PROGRESS_BARS: '1',
      TOKENIZERS_PARALLELISM: 'false',
    };
    this.kind = command;
    this.finishing = false;
    this.exited = new Promise((resolve) => {
      let sawTerminal = false;
      const child = spawn(resolvePython(pythonPath), args, { env, windowsHide: true, shell: false });
      this.child = child;
      let finished = false;
      const finish = (code, message) => {
        if (finished) return;
        finished = true;
        clearTimeout(this.killTimer);
        this.child = null;
        this.kind = null;
        this.finishing = false;
        fs.rmSync(dir, { recursive: true, force: true });
        if (!sawTerminal && command !== 'info') {
          this.onEvent({ event: 'error', message: message || `Engine exited unexpectedly (code ${code}).` });
        }
        if (command === 'info' && !sawTerminal) this.onEvent({ event: 'info_error', message: message || `Engine exited (code ${code}).` });
        resolve(code);
      };
      child.stdout.on('data', lineSplitter((line) => {
        try {
          const ev = JSON.parse(line);
          if (ev && typeof ev.event === 'string') {
            if (['done', 'error', 'info'].includes(ev.event)) { sawTerminal = true; this.finishing = true; }
            this.onEvent(ev);
            return;
          }
        } catch { /* not protocol: fall through to log */ }
        this.onEvent({ event: 'log', message: line });
      }));
      let tail = [];
      child.stderr.on('data', lineSplitter((line) => {
        tail = [...tail.slice(-19), line];
        this.onEvent({ event: 'stderr', message: line });
      }));
      child.on('error', (err) => finish(-1, `Could not start Python (${resolvePython(pythonPath)}): ${err.message}`));
      child.on('close', (code) => finish(code, code ? tail.slice(-6).join('\n') : undefined));
    });
    return this.exited;
  }

  /** Ask a training job to stop after the current step (adapter is still saved). Falls back to a hard kill. */
  cancel(graceMs = 20000) {
    if (!this.child) return false;
    if (this.stopFile) {
      try { fs.writeFileSync(this.stopFile, 'stop'); } catch { /* temp dir already gone */ }
    }
    const child = this.child;
    this.killTimer = setTimeout(() => killTree(child), graceMs);
    return true;
  }

  kill() { killTree(this.child); }
}

module.exports = { Engine, lineSplitter, resolvePython, COMMANDS, REPO_ROOT };
