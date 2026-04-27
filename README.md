# Personal AI MVP v1.5

Locale-first RAG + Graph Knowledge Base:
- Upload PDF/TXT
- Chunking + embedding su ChromaDB
- Estrazione entita/relazioni su SQLite
- Query ibride (RAG + Graph)

## Avvio rapido

```bash
docker-compose up --build
```

Apri: `http://localhost:8000`

## Endpoint principali

- `POST /upload` file PDF/TXT
- `POST /query` body: `{"question":"...", "top_k":4}`
- `GET /graph`
- `GET /export/json`
- `GET /export/csv`

## Note runtime

- Richiede Ollama in esecuzione locale con modello `gemma4:e4b`.
- Se Ollama non e disponibile, il sistema usa fallback euristico per il graph e risponde con contesto grezzo.

## Desktop v2 (Electron onboarding)

Nel percorso `desktop/` trovi un launcher desktop con onboarding guidato:
- check backend
- check/install Ollama (Windows via winget)
- pull modello `gemma4:e4b`
- apertura automatica dell'app web

Comandi:

```bash
cd desktop
npm install
npm start
```

Il backend FastAPI viene avviato automaticamente dal launcher.

### Packaging `.exe` (Windows)

Da `desktop/`:

```bash
npm install
npm run dist:win
```

Output installer:
- `desktop/dist/*.exe`

Nota:
- Il launcher desktop include onboarding automatico Ollama.
- Per eseguire il backend, sulla macchina utente deve essere disponibile Python nel `PATH` (step successivo consigliato: bundle runtime Python dedicato).

### Branding desktop

- Icona app/installer: `favicon (7)/favicon.ico`
- Publisher metadata: campo `author` in `desktop/package.json`

## Preparazione repo pubblica

Checklist gia completata:
- licenza MIT aggiunta (`LICENSE`)
- build artifacts esclusi (`desktop/dist/`, `desktop/node_modules/`)
- metadata desktop aggiornati per distribuzione

Comandi push base:

```bash
git init
git add .
git commit -m "feat: add Personal AI desktop onboarding and windows packaging"
git branch -M main
git remote add origin https://github.com/<tuo-utente>/<tuo-repo>.git
git push -u origin main
```
