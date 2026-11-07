'use strict';
// Renders build/icon.png (512x512). Run: node node_modules/electron/cli.js build/make-icon.js
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
app.disableHardwareAcceleration();
app.whenReady().then(async () => {
  const win = new BrowserWindow({ width: 512, height: 512, show: false, frame: false, transparent: true, webPreferences: { offscreen: true } });
  const html = `<body style="margin:0;background:transparent"><div style="width:512px;height:512px;border-radius:112px;background:linear-gradient(135deg,#f6a13a,#d9690d);display:flex;align-items:center;justify-content:center;font-size:330px;font-family:'Segoe UI Emoji','Apple Color Emoji',sans-serif">🦘</div></body>`;
  await win.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html));
  await new Promise((r) => setTimeout(r, 600));
  const img = await win.webContents.capturePage();
  fs.writeFileSync(path.join(__dirname, 'icon.png'), img.toPNG());
  app.quit();
});
