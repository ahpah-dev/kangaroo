'use strict';
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
// The in-app installer creates its private Python environment under the user-data folder.
process.env.KANGAROO_HOME = process.env.KANGAROO_HOME || app.getPath('userData');
const { Engine } = require('./engine');
const { installEngine } = require('./setup');

const SMOKE = process.argv.includes('--smoke');
let win = null;

const engine = new Engine((ev) => {
  if (win && !win.isDestroyed()) win.webContents.send('engine:event', ev);
});

function createWindow() {
  win = new BrowserWindow({
    width: 1240,
    height: 860,
    minWidth: 980,
    minHeight: 640,
    backgroundColor: '#0f1115',
    title: 'Kangaroo',
    autoHideMenuBar: true,
    show: !SMOKE,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  win.setMenuBarVisibility(false);
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', (e) => e.preventDefault());
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

ipcMain.handle('engine:start', async (_e, command, config, python) => {
  if (typeof command !== 'string') throw new Error('bad command');
  // A job whose final event was already sent (or the startup `info` probe) is just shutting down: wait for it.
  if (engine.running && (engine.kind === 'info' || engine.finishing)) await engine.exited;
  if (engine.running) return { ok: false, error: 'A job is already running.' };
  engine.start(command, config && typeof config === 'object' ? config : {}, typeof python === 'string' ? python : '');
  return { ok: true };
});
let setupRunning = false;
ipcMain.handle('setup:start', (_e, choice) => {
  if (setupRunning || engine.running) return { ok: false, error: 'Another job is running.' };
  const torch = ['auto', 'cuda', 'cpu'].includes(choice) ? choice : 'auto';
  setupRunning = true;
  installEngine((ev) => { if (win && !win.isDestroyed()) win.webContents.send('engine:event', ev); }, torch)
    .finally(() => { setupRunning = false; });
  return { ok: true };
});
ipcMain.handle('engine:cancel', () => engine.cancel());
ipcMain.handle('engine:running', () => engine.running);

ipcMain.handle('dialog:pick', async (_e, kind, options = {}) => {
  const properties = kind === 'dir' ? ['openDirectory', 'createDirectory'] : ['openFile'];
  const filters = Array.isArray(options.filters) ? options.filters : undefined;
  const res = await dialog.showOpenDialog(win, { properties, filters });
  return res.canceled ? null : res.filePaths[0];
});

ipcMain.handle('shell:reveal', (_e, target) => {
  if (typeof target !== 'string' || !fs.existsSync(target)) return false;
  shell.showItemInFolder(path.resolve(target));
  return true;
});

app.whenReady().then(() => {
  createWindow();
  if (SMOKE) runSmoke();
});

app.on('window-all-closed', () => {
  engine.kill();
  app.quit();
});
app.on('before-quit', () => engine.kill());

// `npm run smoke`: load the UI, wait for the engine `info` round-trip, grab a screenshot, exit non-zero on failure.
async function runSmoke() {
  const shot = process.env.KANGAROO_SMOKE_SHOT || path.join(__dirname, 'smoke.png');
  const fail = (msg) => { console.log(JSON.stringify({ smoke: 'fail', error: msg })); app.exit(1); };
  const timer = setTimeout(() => fail('timeout waiting for engine info'), 90000);
  win.webContents.on('console-message', (_e, _lvl, message) => console.log('[renderer]', message));
  await new Promise((r) => win.webContents.once('did-finish-load', r));
  try {
    const deadline = Date.now() + 80000;
    let state;
    while (Date.now() < deadline) {
      state = await win.webContents.executeJavaScript('window.__kangarooState && window.__kangarooState()');
      if (state && state.infoLoaded) break;
      await new Promise((r) => setTimeout(r, 500));
    }
    if (!state || !state.infoLoaded) return fail('renderer never reported engine info');
    // Optional end-to-end: KANGAROO_SMOKE_E2E='{"train":{...},"test":{...},"export":{...}}' drives the real UI.
    if (process.env.KANGAROO_SMOKE_E2E) {
      const plan = JSON.parse(process.env.KANGAROO_SMOKE_E2E);
      const js = (code) => win.webContents.executeJavaScript(code);
      const waitIdle = async (label, ms = 300000) => {
        const end = Date.now() + ms;
        const startBy = Date.now() + 30000;
        // wait for the job to actually start (UI may wait on the previous process to exit)
        while (Date.now() < startBy && !(await js('window.__kangarooState().busy'))) {
          await new Promise((r) => setTimeout(r, 100));
        }
        while (Date.now() < end) {
          const s = await js('window.__kangarooState()');
          if (!s.busy) return s;
          await new Promise((r) => setTimeout(r, 500));
        }
        throw new Error('timeout during ' + label);
      };
      await js(`window.__kangarooFill('train', ${JSON.stringify(plan.train)}); window.__kangarooClick('startBtn')`);
      state = await waitIdle('train');
      if (state.resultBad || !state.resultText) return fail('training failed in UI: ' + state.resultText + state.status);
      const adapter = plan.train.output_dir.replace(/[\\/]+$/, '') + '/adapter';
      await js(`window.__kangarooFill('test', ${JSON.stringify({ ...plan.test, adapter })}); window.__kangarooClick('genBtn')`);
      state = { ...state, ...(await waitIdle('generate')) };
      await js(`window.__kangarooFill('export', ${JSON.stringify({ ...plan.export, adapter })}); window.__kangarooClick('exportBtn')`);
      state = { ...state, ...(await waitIdle('export')) };
      state.e2e = true;
    }
    const tabs = process.env.KANGAROO_SMOKE_TAB;
    if (tabs) await win.webContents.executeJavaScript(`window.__kangarooShowTab(${JSON.stringify(tabs)})`);
    await new Promise((r) => setTimeout(r, 300));
    const image = await win.webContents.capturePage();
    fs.writeFileSync(shot, image.toPNG());
    clearTimeout(timer);
    console.log(JSON.stringify({ smoke: 'ok', state, screenshot: shot }));
    app.exit(0);
  } catch (err) {
    fail(String(err));
  }
}
