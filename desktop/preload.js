const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("personalAI", {
  statusAll: () => ipcRenderer.invoke("status:all"),
  startBackend: () => ipcRenderer.invoke("backend:start"),
  openWebApp: () => ipcRenderer.invoke("app:open-web"),
  installOllamaWin: () => ipcRenderer.invoke("ollama:install-win"),
  openOllamaDownloadPage: () => ipcRenderer.invoke("ollama:open-download-page"),
  pullModel: () => ipcRenderer.invoke("ollama:pull-model"),
  onPullLog: (cb) => ipcRenderer.on("ollama:pull-log", (_evt, log) => cb(log))
});
