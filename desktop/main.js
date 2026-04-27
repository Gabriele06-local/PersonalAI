const { app, BrowserWindow, ipcMain, shell } = require("electron");
const { spawn, exec } = require("child_process");
const path = require("path");

const BACKEND_HOST = "127.0.0.1";
const BACKEND_PORT = 8000;
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const OLLAMA_URL = "http://127.0.0.1:11434";
const MODEL_NAME = "gemma4:e4b";

let mainWindow;
let backendProcess = null;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function isBackendReady() {
  try {
    await fetchJson(`${BACKEND_URL}/health`);
    return true;
  } catch {
    return false;
  }
}

async function isOllamaReady() {
  try {
    await fetchJson(`${OLLAMA_URL}/api/tags`);
    return true;
  } catch {
    return false;
  }
}

function startBackend() {
  if (backendProcess) return;

  const rootDir = app.isPackaged
    ? path.join(process.resourcesPath, "backend")
    : path.resolve(__dirname, "..");
  const dataDir = path.join(app.getPath("userData"), "data");
  const pythonExe = process.env.PERSONAL_AI_PYTHON || "python";
  backendProcess = spawn(
    pythonExe,
    ["-m", "uvicorn", "app:app", "--host", BACKEND_HOST, "--port", String(BACKEND_PORT)],
    {
      cwd: rootDir,
      windowsHide: true,
      env: {
        ...process.env,
        PERSONAL_AI_DATA_DIR: dataDir
      }
    }
  );

  backendProcess.on("exit", () => {
    backendProcess = null;
  });
}

function stopBackend() {
  if (!backendProcess) return;
  backendProcess.kill();
  backendProcess = null;
}

async function waitBackend(maxMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < maxMs) {
    if (await isBackendReady()) return true;
    await sleep(700);
  }
  return false;
}

async function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1180,
    height: 780,
    title: "Personal AI",
    webPreferences: {
      preload: path.join(__dirname, "preload.js")
    }
  });

  await mainWindow.loadFile(path.join(__dirname, "onboarding.html"));
}

ipcMain.handle("status:all", async () => {
  const ollama = await isOllamaReady();
  const backend = await isBackendReady();
  return { ollama, backend, model: MODEL_NAME };
});

ipcMain.handle("backend:start", async () => {
  if (!(await isBackendReady())) {
    startBackend();
    await waitBackend(20000);
  }
  return { ok: await isBackendReady(), url: BACKEND_URL };
});

ipcMain.handle("app:open-web", async () => {
  if (!(await isBackendReady())) {
    startBackend();
    await waitBackend(20000);
  }
  if (await isBackendReady()) {
    await mainWindow.loadURL(BACKEND_URL);
    return { ok: true };
  }
  return { ok: false, error: "Backend non raggiungibile" };
});

ipcMain.handle("ollama:open-download-page", async () => {
  await shell.openExternal("https://ollama.com/download/windows");
  return { ok: true };
});

ipcMain.handle("ollama:install-win", async () => {
  if (process.platform !== "win32") {
    return { ok: false, error: "Install automatica disponibile solo su Windows" };
  }

  return new Promise((resolve) => {
    const cmd =
      "winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements";
    exec(cmd, { windowsHide: true }, (error, stdout, stderr) => {
      if (error) {
        resolve({ ok: false, error: stderr || error.message || "Installazione fallita" });
        return;
      }
      resolve({ ok: true, output: stdout || "Installazione completata" });
    });
  });
});

ipcMain.handle("ollama:pull-model", async () => {
  return new Promise((resolve) => {
    const proc = spawn("ollama", ["pull", MODEL_NAME], { windowsHide: true });
    let logs = "";

    proc.stdout.on("data", (data) => {
      logs += data.toString();
      if (mainWindow) {
        mainWindow.webContents.send("ollama:pull-log", data.toString());
      }
    });
    proc.stderr.on("data", (data) => {
      logs += data.toString();
      if (mainWindow) {
        mainWindow.webContents.send("ollama:pull-log", data.toString());
      }
    });
    proc.on("close", (code) => {
      resolve({
        ok: code === 0,
        code,
        output: logs || "Nessun log disponibile"
      });
    });
  });
});

app.whenReady().then(async () => {
  await createMainWindow();
  startBackend();
});

app.on("before-quit", () => {
  stopBackend();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

