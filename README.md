# Personal AI - MVP v1.5

`Personal AI` e un assistente locale che unisce due approcci:
- **RAG vettoriale** (recupero contesto da documenti)
- **Knowledge Graph** (entita e relazioni salvate nel tempo)

In pratica: carichi file, fai domande, e ottieni risposte con contesto + collegamenti tra concetti.

## Cosa fa oggi

### 1) Upload e indicizzazione documenti
- Accetta file `PDF` e `TXT`
- Estrae testo (PyMuPDF per PDF)
- Divide il testo in chunk
- Crea embedding su ChromaDB persistente
- Estrae entita/relazioni e le salva in SQLite

### 2) Query ibrida
- Endpoint domanda: combina risultati RAG + relazioni graph
- Se Ollama e disponibile, genera una risposta LLM completa
- Se Ollama non e disponibile, usa fallback e restituisce comunque contesto utile

### 3) Storico conoscenza
- Il graph viene salvato su database locale (`SQLite`)
- Ogni relazione ha timestamp e sorgente documento
- E possibile esportare relazioni in JSON/CSV

### 4) UI Web pronta
- Interfaccia su `http://127.0.0.1:8000`
- Drag&drop file
- Query testuale
- Snapshot graph + export
- Graph Visualizer interattivo (Cytoscape)
- Path Finder (`start -> end`) con highlight e zoom
- Toggle grafi diretti/indiretti
- Presets path salvabili con tag e pin

### 5) Freemium base
- Limite piano Free su upload (`FREE_UPLOAD_LIMIT`, default 10)
- Banner upgrade quando uso supera l'80%
- Endpoint usage e upgrade per gestione stato piano

### 6) Launcher Desktop (Electron)
- Presente in `desktop/`
- Onboarding con:
  - check backend
  - check/install Ollama (Windows via winget)
  - pull modello `gemma4:e4b`
  - apertura app automatica

---

## Stack attuale

- **Backend:** FastAPI (`app.py`)
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB:** ChromaDB (persistente)
- **Graph DB:** SQLite (`entities`, `relations`)
- **LLM locale:** Ollama (`gemma4:e4b`)
- **Desktop wrapper:** Electron + electron-builder

---

## Avvio rapido (sviluppo locale)

### Modalita Python
```bash
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Apri poi: `http://127.0.0.1:8000`

### Modalita Desktop
```bash
cd desktop
npm install
npm start
```

---

## Packaging Windows (.exe)

Da `desktop/`:

```bash
npm run dist:win
```

Output:
- `desktop/dist/Personal AI Setup <versione>.exe`

Icona app/installer:
- `favicon (7)/favicon.ico`

---

## API principali

- `POST /upload` -> upload PDF/TXT
- `POST /query` -> body `{"question":"...", "top_k":4}`
- `GET /graph` -> entita + relazioni recenti
- `GET /graph/nodes` -> nodi per visualizer
- `GET /graph/edges` -> archi per visualizer
- `GET /graph/path?start=A&end=B&undirected=true|false` -> percorso nel graph
- `POST /path/save` -> salva percorso frequente
- `GET /paths/presets` -> preset default + salvati (tag/pin)
- `GET /usage` -> stato piano freemium/pro
- `GET /export/json` -> export graph JSON
- `GET /export/csv` -> export relazioni CSV
- `GET /health` -> stato servizio

---

## Limiti attuali (MVP)

- Nessuna autenticazione utenti (single-user locale)
- Possibili duplicati se si ricarica lo stesso file piu volte
- Per risposta LLM completa serve Ollama attivo in locale
- Su desktop e richiesto Python disponibile nel `PATH`
- Endpoint `POST /upgrade` usa secret header (`x-upgrade-secret`) pensato per ambiente dev/demo

---

## Licenza

MIT. Vedi file `LICENSE`.
