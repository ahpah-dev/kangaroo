'use strict';
// First-run engine installer: creates a private venv and installs PyTorch + dependencies.
// Electron-free so it can be tested with plain node.
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const { lineSplitter, venvPython, managedVenvDir, REQUIREMENTS } = require('./engine');

const MIN_PYTHON = [3, 10];
const TORCH_INDEX = { cuda: 'https://download.pytorch.org/whl/cu128', cpu: 'https://download.pytorch.org/whl/cpu' };

function pythonVersion(cmd, args = []) {
  const r = spawnSync(cmd, [...args, '-c', 'import sys;print("%d.%d.%d"%sys.version_info[:3])'], { encoding: 'utf8', windowsHide: true });
  if (r.status !== 0) return null;
  const v = r.stdout.trim().split('.').map(Number);
  return v.length === 3 && v.every(Number.isFinite) ? v : null;
}

/** Find a system Python >= 3.10. Returns { cmd, args, version } or null. */
function findSystemPython() {
  const candidates = process.platform === 'win32'
    ? [['py', ['-3']], ['python', []], ['python3', []]]
    : [['python3', []], ['python', []]];
  for (const [cmd, args] of candidates) {
    const v = pythonVersion(cmd, args);
    if (v && (v[0] > MIN_PYTHON[0] || (v[0] === MIN_PYTHON[0] && v[1] >= MIN_PYTHON[1]))) return { cmd, args, version: v.join('.') };
  }
  return null;
}

function hasNvidiaGpu() {
  const r = spawnSync('nvidia-smi', ['-L'], { encoding: 'utf8', windowsHide: true });
  return r.status === 0 && /GPU \d+/.test(r.stdout || '');
}

function run(cmd, args, onLine, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true, shell: false, env: { ...process.env, PIP_DISABLE_PIP_VERSION_CHECK: '1', PYTHONUTF8: '1', ...env } });
    const feed = lineSplitter(onLine);
    child.stdout.on('data', feed);
    child.stderr.on('data', feed);
    child.on('error', reject);
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} ${args.slice(0, 3).join(' ')} … exited with code ${code}`))));
  });
}

/**
 * @param {(ev:object)=>void} emit  receives {event:'setup_step'|'setup_log'|'setup_done'|'setup_error', ...}
 * @param {'auto'|'cuda'|'cpu'} torchChoice
 */
async function installEngine(emit, torchChoice = 'auto') {
  const log = (message) => emit({ event: 'setup_log', message });
  const step = (name) => emit({ event: 'setup_step', name });
  try {
    const dir = managedVenvDir();
    if (!dir) throw new Error('No application data folder available.');
    step('Looking for Python 3.10+');
    const sys = findSystemPython();
    if (!sys) throw new Error('Python 3.10 or newer was not found. Install it from python.org (tick "Add python.exe to PATH"), then try again.');
    log(`Using Python ${sys.version}`);

    const choice = torchChoice === 'auto' ? (hasNvidiaGpu() ? 'cuda' : 'cpu') : torchChoice;
    log(choice === 'cuda' ? 'NVIDIA GPU found: installing CUDA 12.8 PyTorch (about 3 GB).' : 'Installing CPU-only PyTorch.');

    if (!fs.existsSync(venvPython(dir))) {
      step('Creating private environment');
      await run(sys.cmd, [...sys.args, '-m', 'venv', dir], log);
    }
    const py = venvPython(dir);
    step('Installing PyTorch (this takes a while)');
    await run(py, ['-m', 'pip', 'install', '--no-input', '--progress-bar', 'off', 'torch', '--index-url', TORCH_INDEX[choice]], log);
    step('Installing Transformers, PEFT and friends');
    await run(py, ['-m', 'pip', 'install', '--no-input', '--progress-bar', 'off', '-r', REQUIREMENTS], log);
    emit({ event: 'setup_done', python: py, torch: choice });
  } catch (err) {
    emit({ event: 'setup_error', message: String(err && err.message ? err.message : err) });
  }
}

module.exports = { installEngine, findSystemPython, hasNvidiaGpu };
