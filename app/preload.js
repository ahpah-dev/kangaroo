'use strict';
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('kangaroo', {
  start: (command, config, python) => ipcRenderer.invoke('engine:start', command, config, python),
  cancel: () => ipcRenderer.invoke('engine:cancel'),
  running: () => ipcRenderer.invoke('engine:running'),
  pick: (kind, options) => ipcRenderer.invoke('dialog:pick', kind, options),
  reveal: (target) => ipcRenderer.invoke('shell:reveal', target),
  onEvent: (cb) => {
    const handler = (_e, ev) => cb(ev);
    ipcRenderer.on('engine:event', handler);
    return () => ipcRenderer.removeListener('engine:event', handler);
  },
});
