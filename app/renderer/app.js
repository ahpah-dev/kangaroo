'use strict';
/* Kangaroo renderer. No frameworks, no innerHTML with dynamic data. */

const $ = (id) => document.getElementById(id);
const api = window.kangaroo;

// ---------------------------------------------------------------- schema
const MODELS = [
  'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct',
  'HuggingFaceTB/SmolLM2-360M-Instruct', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'microsoft/Phi-3.5-mini-instruct',
];
const DATA_FILTERS = [{ name: 'Datasets', extensions: ['jsonl', 'json', 'csv', 'tsv'] }];

// Keep defaults in sync with src/kangaroo/config.py (checked by tests/test_defaults.py).
const TRAIN_GROUPS = [
  { title: 'Model & data', fields: [
    { key: 'model', label: 'Base model', type: 'text', def: 'Qwen/Qwen2.5-0.5B-Instruct', full: true, list: 'models', hint: 'Hugging Face id or local folder' },
    { key: 'dataset', label: 'Dataset', type: 'path', def: '', full: true, pick: 'file', filters: DATA_FILTERS, hint: '.jsonl / .json / .csv, or a Hub dataset id' },
    { key: 'output_dir', label: 'Output folder', type: 'path', def: 'kangaroo-out', full: true, pick: 'dir' },
    { key: 'dataset_format', label: 'Format', type: 'select', def: 'auto', options: ['auto', 'alpaca', 'chat', 'prompt_completion', 'text'] },
    { key: 'max_seq_len', label: 'Max sequence length', type: 'number', def: 2048, min: 16, step: 1 },
    { key: 'val_fraction', label: 'Validation split', type: 'number', def: 0, min: 0, max: 0.49, step: 0.01 },
    { key: 'train_on_inputs', label: 'Also train on prompts', type: 'bool', def: false },
  ] },
  { title: 'LoRA', fields: [
    { key: 'lora_r', label: 'Rank (r)', type: 'number', def: 16, min: 1, step: 1 },
    { key: 'lora_alpha', label: 'Alpha', type: 'number', def: 32, min: 1, step: 1 },
    { key: 'lora_dropout', label: 'Dropout', type: 'number', def: 0, min: 0, max: 0.9, step: 0.01 },
    { key: 'load_in_4bit', label: '4-bit base (QLoRA, needs CUDA)', type: 'bool', def: false },
    { key: 'rslora', label: 'Rank-stabilised scaling (rsLoRA)', type: 'bool', def: true },
  ] },
  { title: 'Optimisation', fields: [
    { key: 'epochs', label: 'Epochs', type: 'number', def: 1, min: 0, step: 0.1 },
    { key: 'max_steps', label: 'Max steps (-1 = use epochs)', type: 'number', def: -1, step: 1 },
    { key: 'learning_rate', label: 'Learning rate', type: 'number', def: 0.0002, min: 0, step: 0.00005 },
    { key: 'lr_scheduler', label: 'Schedule', type: 'select', def: 'cosine', options: ['cosine', 'linear', 'constant'] },
    { key: 'batch_tokens', label: 'Tokens per micro-batch', type: 'number', def: 4096, min: 64, step: 256, hint: 'Padded-token budget; raise until VRAM is full' },
    { key: 'grad_accum', label: 'Gradient accumulation', type: 'number', def: 4, min: 1, step: 1 },
    { key: 'warmup_ratio', label: 'Warmup ratio', type: 'number', def: 0.03, min: 0, max: 0.9, step: 0.01 },
    { key: 'weight_decay', label: 'Weight decay', type: 'number', def: 0, min: 0, step: 0.01 },
  ] },
  { title: 'Speed & memory', fields: [
    { key: 'gradient_checkpointing', label: 'Activation checkpointing', type: 'select', def: 'auto', options: ['auto', 'on', 'off'], hint: 'auto = off if the biggest batch fits in VRAM' },
    { key: 'loss_chunk', label: 'Loss chunk (tokens, 0 = off)', type: 'number', def: 1024, min: 0, step: 128, hint: 'Avoids materialising full-vocab logits' },
    { key: 'precision', label: 'Precision', type: 'select', def: 'auto', options: ['auto', 'bf16', 'fp16', 'fp32'] },
    { key: 'seed', label: 'Seed', type: 'number', def: 3407, step: 1 },
  ] },
];
const TEST_FIELDS = [
  { key: 'adapter', label: 'Adapter folder', type: 'path', def: '', pick: 'dir', full: true },
  { key: 'system', label: 'System prompt (optional)', type: 'text', def: '', full: true },
  { key: 'prompt', label: 'Prompt', type: 'textarea', def: '', full: true },
  { key: 'temperature', label: 'Temperature (0 = greedy)', type: 'number', def: 0.7, min: 0, max: 2, step: 0.1 },
  { key: 'max_new_tokens', label: 'Max new tokens', type: 'number', def: 256, min: 1, step: 1 },
];
const EXPORT_FIELDS = [
  { key: 'adapter', label: 'Adapter folder', type: 'path', def: '', pick: 'dir', full: true },
  { key: 'output_dir', label: 'Output folder', type: 'path', def: 'kangaroo-merged', pick: 'dir', full: true },
  { key: 'precision', label: 'Precision', type: 'select', def: 'bf16', options: ['bf16', 'fp16', 'fp32'] },
];
const SETTINGS_FIELDS = [
  { key: 'python', label: 'Python interpreter', type: 'path', def: '', pick: 'file', full: true, hint: 'Leave empty to use the engine Kangaroo installed, then the project .venv, then PATH' },
];

// ---------------------------------------------------------------- form building
const store = {
  load(ns) { try { return JSON.parse(localStorage.getItem('kangaroo.' + ns) || '{}'); } catch { return {}; } },
  save(ns, v) { localStorage.setItem('kangaroo.' + ns, JSON.stringify(v)); },
};

function el(tag, props = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node[k] = v;
  }
  for (const kid of kids) if (kid) node.append(kid);
  return node;
}

const inputs = {}; // ns -> key -> input

function buildField(ns, f, saved) {
  const id = `${ns}-${f.key}`;
  const value = saved[f.key] !== undefined ? saved[f.key] : f.def;
  let input;
  if (f.type === 'select') {
    input = el('select', { id }, ...f.options.map((o) => el('option', { value: o, text: o })));
    input.value = value;
  } else if (f.type === 'bool') {
    input = el('input', { id, type: 'checkbox', checked: !!value });
  } else if (f.type === 'textarea') {
    input = el('textarea', { id, value, rows: 4 });
  } else if (f.type === 'number') {
    input = el('input', { id, type: 'number', value });
    for (const a of ['min', 'max', 'step']) if (f[a] !== undefined) input.setAttribute(a, f[a]);
    input.setAttribute('step', f.step !== undefined ? f.step : 'any');
  } else {
    input = el('input', { id, type: 'text', value, spellcheck: false });
    if (f.list) input.setAttribute('list', f.list);
  }
  (inputs[ns] = inputs[ns] || {})[f.key] = input;
  input.addEventListener('change', () => store.save(ns, readForm(ns, true)));

  let control = input;
  if (f.pick) {
    const btn = el('button', { class: 'btn small', type: 'button', text: 'Browse…' });
    btn.addEventListener('click', async () => {
      const p = await api.pick(f.pick, { filters: f.filters });
      if (p) { input.value = p; input.dispatchEvent(new Event('change')); }
    });
    control = el('div', { class: 'row' }, input, btn);
  }
  if (f.type === 'bool') {
    return el('div', { class: 'field check full' }, input, el('label', { htmlFor: id, text: f.label }));
  }
  return el('div', { class: 'field' + (f.full ? ' full' : '') },
    el('label', { htmlFor: id, text: f.label }), control, f.hint ? el('div', { class: 'hint', text: f.hint }) : null);
}

function readForm(ns, raw = false) {
  const out = {};
  for (const [key, input] of Object.entries(inputs[ns])) {
    if (input.type === 'checkbox') out[key] = input.checked;
    else if (input.type === 'number') { const n = input.value === '' ? NaN : Number(input.value); if (!Number.isNaN(n)) out[key] = n; else if (raw) out[key] = input.value; }
    else out[key] = input.value.trim();
  }
  return out;
}

function renderSimpleForm(ns, container, fields) {
  const saved = store.load(ns);
  const grid = el('div', { class: 'grid2' }, ...fields.map((f) => buildField(ns, f, saved)));
  container.append(grid);
}

function renderTrainForm() {
  const saved = store.load('train');
  const host = $('trainForm');
  host.append(el('datalist', { id: 'models' }, ...MODELS.map((m) => el('option', { value: m }))));
  for (const g of TRAIN_GROUPS) {
    host.append(el('div', { class: 'card' }, el('h3', { text: g.title }),
      el('div', { class: 'grid2' }, ...g.fields.map((f) => buildField('train', f, saved)))));
  }
}

// ---------------------------------------------------------------- tabs
function showTab(name) {
  document.querySelectorAll('.tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + name));
  if (name === 'train') requestAnimationFrame(drawChart);
}
document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
window.__kangarooShowTab = showTab;

// ---------------------------------------------------------------- helpers
const fmt = {
  int: (n) => (n == null ? '–' : Math.round(n).toLocaleString()),
  num: (n, d = 4) => (n == null || !Number.isFinite(n) ? '–' : n.toFixed(d)),
  dur: (s) => {
    if (s == null || !Number.isFinite(s)) return '–';
    s = Math.round(s);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
  },
  bytes: (n) => (n > 2 ** 30 ? (n / 2 ** 30).toFixed(2) + ' GB' : (n / 2 ** 20).toFixed(1) + ' MB'),
};

function logLine(text, cls = '') {
  const log = $('log');
  const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
  log.append(el('div', { class: cls, text }));
  while (log.childElementCount > 1500) log.firstChild.remove();
  if (stick) log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------- chart
let history = [];
function ema(values, a = 0.15) {
  const out = []; let s = null;
  for (const v of values) { s = s == null ? v : a * v + (1 - a) * s; out.push(s); }
  return out;
}
function drawChart() {
  const canvas = $('chart');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  $('chartEmpty').classList.toggle('hidden', history.length > 1);
  if (history.length < 2) return;
  const css = getComputedStyle(document.documentElement);
  const pad = { l: 48, r: 14, t: 14, b: 24 };
  const ys = history.map((p) => p.loss);
  const sm = ema(ys);
  let lo = Math.min(...sm, ...ys.map((y, i) => (ys.length > 60 ? sm[i] : y)));
  let hi = Math.max(...ys.slice(0, 1), ...sm);
  hi = Math.max(hi, ...ys.filter((_, i) => i > ys.length * 0.02));
  if (hi === lo) hi = lo + 1;
  const x0 = history[0].step, x1 = history[history.length - 1].step || 1;
  const X = (s) => pad.l + ((s - x0) / Math.max(1, x1 - x0)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  ctx.strokeStyle = css.getPropertyValue('--line'); ctx.fillStyle = css.getPropertyValue('--muted'); ctx.font = '11px sans-serif';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), 6, y + 4);
  }
  ctx.fillText(String(x0), pad.l, h - 6); ctx.fillText('step ' + x1, w - pad.r - 54, h - 6);
  const line = (vals, color, width) => {
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width;
    vals.forEach((v, i) => { const x = X(history[i].step), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  };
  line(ys, 'rgba(240,138,36,.28)', 1);
  line(sm, css.getPropertyValue('--accent'), 2);
}
window.addEventListener('resize', drawChart);

// ---------------------------------------------------------------- job state
let job = null; // { kind }
const pythonPath = () => (inputs.settings && inputs.settings.python.value.trim()) || '';

function setBusy(busy) {
  document.querySelectorAll('#trainForm input, #trainForm select, #testForm input, #testForm textarea, #testForm select, #exportForm input, #exportForm select')
    .forEach((n) => { n.disabled = busy; });
  document.querySelectorAll('#trainForm .btn, #testForm .btn, #exportForm .btn').forEach((n) => { n.disabled = busy; });
  $('startBtn').disabled = busy; $('genBtn').disabled = busy; $('exportBtn').disabled = busy;
  $('stopBtn').disabled = !(busy && job && job.kind === 'train');
}

async function launch(kind, command, config, statusEl) {
  const res = await api.start(command, config, pythonPath());
  if (!res.ok) { (statusEl || $('status')).textContent = res.error; return false; }
  job = { kind };
  setBusy(true);
  return true;
}

function finishJob() { job = null; setBusy(false); }

// ---------------------------------------------------------------- train
function resetRun() {
  history = []; drawChart();
  $('log').replaceChildren();
  $('result').hidden = true; $('result').classList.remove('bad');
  $('bar').style.width = '0';
  for (const id of ['tStep', 'tLoss', 'tTok', 'tMem', 'tEta']) $(id).textContent = '–';
}

$('startBtn').addEventListener('click', async () => {
  const cfg = readForm('train');
  if (!cfg.model || !cfg.dataset) { $('status').textContent = 'Choose a base model and a dataset first.'; return; }
  if (!cfg.val_fraction) delete cfg.val_fraction;
  resetRun();
  $('status').textContent = 'Starting…';
  if (!(await launch('train', 'train', cfg))) $('status').textContent = 'Idle';
});
$('stopBtn').addEventListener('click', async () => {
  $('stopBtn').disabled = true;
  $('status').textContent = 'Stopping after the current step — the adapter will be saved…';
  await api.cancel();
});

function showResult(box, lines, pathText, buttons = [], bad = false) {
  box.replaceChildren(...lines.map((t) => el('div', { text: t })));
  if (pathText) box.append(el('div', { class: 'path', text: pathText }));
  if (buttons.length) box.append(el('div', { class: 'row' }, ...buttons));
  box.classList.toggle('bad', bad);
  box.hidden = false;
}

function onTrainEvent(ev) {
  switch (ev.event) {
    case 'phase': $('status').textContent = { data: 'Preparing dataset…', model: 'Loading model…', train: 'Training' }[ev.name] || ev.name; break;
    case 'data': logLine(`Dataset: ${ev.examples.toLocaleString()} examples (${ev.format}), ${ev.tokens.toLocaleString()} tokens, ${ev.target_tokens.toLocaleString()} supervised`, 'dim'); break;
    case 'plan': logLine(`Plan: ${ev.total_steps} steps, padding efficiency ${(ev.padding_efficiency * 100).toFixed(1)}%`, 'dim'); break;
    case 'checkpointing': logLine(`Activation checkpointing: ${ev.mode}`, 'dim'); break;
    case 'progress': {
      history.push(ev);
      $('tStep').textContent = `${ev.step}/${ev.total_steps}`;
      $('tLoss').textContent = fmt.num(ev.loss);
      $('tTok').textContent = fmt.int(ev.tok_s);
      $('tMem').textContent = ev.mem_gb == null ? 'CPU' : ev.mem_gb.toFixed(2) + ' GB';
      $('tEta').textContent = fmt.dur(ev.eta_s);
      $('bar').style.width = (100 * ev.step / ev.total_steps).toFixed(1) + '%';
      $('status').textContent = `Training — epoch ${ev.epoch + 1}`;
      logLine(`step ${ev.step}/${ev.total_steps}  loss ${fmt.num(ev.loss)}  lr ${ev.lr.toExponential(2)}  grad ${fmt.num(ev.grad_norm, 2)}  ${fmt.int(ev.tok_s)} tok/s`);
      drawChart();
      break;
    }
    case 'eval': logLine(`validation loss ${fmt.num(ev.val_loss)}`, 'good'); break;
    case 'done': {
      $('status').textContent = ev.stopped ? 'Stopped — adapter saved' : 'Finished';
      const lines = [
        ev.stopped ? `Stopped at step ${ev.steps}/${ev.total_steps}.` : `Trained ${ev.steps} steps in ${fmt.dur(ev.train_seconds)}.`,
        `Final loss ${fmt.num(ev.final_loss)}${ev.val_loss != null ? `, validation ${fmt.num(ev.val_loss)}` : ''} · ${fmt.int(ev.avg_tok_s)} tokens/s · peak VRAM ${ev.peak_mem_gb == null ? 'n/a (CPU)' : ev.peak_mem_gb.toFixed(2) + ' GB'}`,
      ];
      const reveal = el('button', { class: 'btn small', text: 'Show in folder' });
      reveal.addEventListener('click', () => api.reveal(ev.adapter_dir));
      const test = el('button', { class: 'btn small', text: 'Test it' });
      test.addEventListener('click', () => { for (const ns of ['test', 'export']) { inputs[ns].adapter.value = ev.adapter_dir; inputs[ns].adapter.dispatchEvent(new Event('change')); } showTab('test'); });
      showResult($('result'), lines, ev.adapter_dir, [reveal, test]);
      logLine('Done.', 'good');
      finishJob();
      break;
    }
    case 'error':
      $('status').textContent = 'Failed';
      showResult($('result'), ['Training failed: ' + ev.message], '', [], true);
      if (ev.traceback) logLine(ev.traceback, 'err');
      finishJob();
      break;
    case 'log': logLine(ev.message); break;
    case 'stderr': logLine(ev.message, 'dim'); break;
    default: break;
  }
}

// ---------------------------------------------------------------- test
$('genBtn').addEventListener('click', async () => {
  const cfg = readForm('test');
  if (!cfg.adapter || !cfg.prompt) { $('genStatus').textContent = 'Pick an adapter and write a prompt.'; return; }
  $('genOut').textContent = '';
  $('genStatus').textContent = 'Loading model…';
  if (!(await launch('test', 'generate', cfg, $('genStatus')))) return;
});
function onTestEvent(ev) {
  if (ev.event === 'token') { $('genStatus').textContent = 'Generating…'; $('genOut').textContent += ev.text; }
  else if (ev.event === 'done') { $('genStatus').textContent = 'Done'; finishJob(); }
  else if (ev.event === 'error') { $('genStatus').textContent = ''; $('genOut').textContent = 'Error: ' + ev.message; finishJob(); }
}

// ---------------------------------------------------------------- export
$('exportBtn').addEventListener('click', async () => {
  const cfg = readForm('export');
  if (!cfg.adapter || !cfg.output_dir) { $('exportStatus').textContent = 'Pick an adapter and an output folder.'; return; }
  $('exportResult').hidden = true;
  $('exportStatus').textContent = 'Working…';
  await launch('export', 'export', cfg, $('exportStatus'));
});
function onExportEvent(ev) {
  if (ev.event === 'log') $('exportStatus').textContent = ev.message;
  else if (ev.event === 'done') {
    $('exportStatus').textContent = '';
    const reveal = el('button', { class: 'btn small', text: 'Show in folder' });
    reveal.addEventListener('click', () => api.reveal(ev.output_dir));
    showResult($('exportResult'), [`Saved merged model (${fmt.bytes(ev.size_bytes)}).`], ev.output_dir, [reveal]);
    finishJob();
  } else if (ev.event === 'error') {
    $('exportStatus').textContent = '';
    showResult($('exportResult'), ['Export failed: ' + ev.message], '', [], true);
    finishJob();
  }
}

// ---------------------------------------------------------------- engine info / settings
let infoLoaded = false, infoOk = false;
function onInfo(ev) {
  infoLoaded = true;
  const dot = $('sysDot');
  if (ev.event === 'info') {
    infoOk = true;
    dot.className = 'sys-dot ' + (ev.cuda ? 'ok' : 'cpu');
    $('sysTitle').textContent = ev.cuda ? ev.gpu : 'CPU only';
    $('sysSub').textContent = ev.cuda ? `${ev.vram_gb} GB VRAM · ${ev.bf16 ? 'bf16' : 'fp16'}` : 'PyTorch has no CUDA';
    const rows = [['Kangaroo', ev.kangaroo], ['Python', ev.python], ['PyTorch', ev.torch], ['Transformers', ev.transformers], ['PEFT', ev.peft],
      ['bitsandbytes', ev.bitsandbytes || 'not installed (needed for 4-bit)'], ['datasets', ev.datasets || 'not installed (needed for Hub datasets)'], ['Platform', ev.platform]];
    $('infoTable').replaceChildren(...rows.map(([k, v]) => el('tr', {}, el('td', { text: k }), el('td', { text: String(v) }))));
    $('setupHelp').hidden = true;
    if (!ev.cuda && ev.note) logLine(ev.note, 'dim');
  } else {
    infoOk = false;
    dot.className = 'sys-dot bad';
    $('sysTitle').textContent = 'Engine not ready';
    $('sysSub').textContent = 'See Settings';
    $('setupMsg').textContent = ev.message || 'The Python engine could not start.';
    $('setupHelp').hidden = false;
    if (!setupBusy) showTab('settings');
    $('infoTable').replaceChildren();
  }
}
function checkEngine() {
  infoLoaded = false;
  $('sysDot').className = 'sys-dot'; $('sysTitle').textContent = 'Checking engine…'; $('sysSub').textContent = '';
  api.start('info', {}, pythonPath()).then((r) => { if (!r.ok) setTimeout(checkEngine, 500); });
}
$('recheckBtn').addEventListener('click', checkEngine);

// ---------------------------------------------------------------- in-app engine installer
let setupBusy = false;
function setupLine(text, cls = '') {
  const box = $('setupLog'); box.hidden = false;
  box.append(el('div', { class: cls, text }));
  while (box.childElementCount > 800) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}
$('setupBtn').addEventListener('click', async () => {
  $('setupLog').replaceChildren();
  const res = await api.setup($('setupTorch').value);
  if (!res.ok) { $('setupStep').textContent = res.error || 'Could not start setup.'; return; }
  setupBusy = true; $('setupBtn').disabled = true; $('setupTorch').disabled = true;
  $('setupStep').textContent = 'Starting…';
});
function onSetupEvent(ev) {
  if (ev.event === 'setup_step') { $('setupStep').textContent = ev.name + '…'; setupLine('> ' + ev.name, 'dim'); }
  else if (ev.event === 'setup_log') setupLine(ev.message);
  else if (ev.event === 'setup_done' || ev.event === 'setup_error') {
    setupBusy = false; $('setupBtn').disabled = false; $('setupTorch').disabled = false;
    if (ev.event === 'setup_done') { $('setupStep').textContent = 'Installed. Checking…'; setupLine('Engine installed.', 'good'); checkEngine(); }
    else { $('setupStep').textContent = 'Setup failed.'; setupLine(ev.message, 'err'); }
  }
}

// ---------------------------------------------------------------- event routing
api.onEvent((ev) => {
  if (ev.event.startsWith('setup_')) { onSetupEvent(ev); return; }
  if (ev.event === 'info' || ev.event === 'info_error') { onInfo(ev); return; }
  if (!job) return;
  if (job.kind === 'train') onTrainEvent(ev);
  else if (job.kind === 'test') onTestEvent(ev);
  else if (job.kind === 'export') onExportEvent(ev);
});

// ---------------------------------------------------------------- boot
renderTrainForm();
renderSimpleForm('test', $('testForm'), TEST_FIELDS);
renderSimpleForm('export', $('exportForm'), EXPORT_FIELDS);
renderSimpleForm('settings', $('settingsForm'), SETTINGS_FIELDS);
setBusy(false);
checkEngine();
// Test hooks used by `npm run smoke` / smoke:train (harmless in normal use).
window.__kangarooFill = (ns, values) => {
  for (const [k, v] of Object.entries(values)) {
    const i = inputs[ns][k];
    if (i.type === 'checkbox') i.checked = !!v; else i.value = v;
  }
};
window.__kangarooClick = (id) => $(id).click();
window.__kangarooState = () => ({
  infoLoaded, infoOk, fields: Object.keys(inputs.train).length,
  busy: job !== null, startDisabled: $('startBtn').disabled, formDisabled: inputs.train.model.disabled, status: $('status').textContent, points: history.length,
  resultText: $('result').hidden ? '' : $('result').textContent, resultBad: $('result').classList.contains('bad'),
  genOut: $('genOut').textContent, genStatus: $('genStatus').textContent,
  exportText: $('exportResult').hidden ? '' : $('exportResult').textContent,
  defaults: Object.fromEntries(Object.entries(inputs.train).map(([k, i]) => [k, i.type === 'checkbox' ? i.checked : i.type === 'number' ? Number(i.value) : i.value])),
  sys: $('sysTitle').textContent,
});
