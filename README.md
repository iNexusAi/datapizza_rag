# 🍕 DataPizza RAG System

Sistema completo di **Retrieval-Augmented Generation (RAG)** costruito con [datapizza.ai](https://datapizza.ai/) e Streamlit.

## 📋 Caratteristiche

- ✅ **Upload di documenti** (PDF e TXT)
- ✅ **Chunking automatico** con overlap
- ✅ **Embedding generation** con modelli OpenAI
- ✅ **Vector storage** con Qdrant (in-memory o server esterno)
- ✅ **Query rewriting** per migliorare il retrieval
- ✅ **Risposta contestuale** basata sui documenti caricati
- ✅ **Interfaccia UI moderna** con Streamlit
- ✅ **Chat history** con visualizzazione delle fonti

## 🚀 Installazione

### 1. Crea e attiva l'ambiente virtuale

```bash
# Crea l'ambiente virtuale con Python 3.12
python3.12 -m venv env_project

# Attiva l'ambiente (macOS/Linux)
source env_project/bin/activate

# Attiva l'ambiente (Windows)
env_project\Scripts\activate
```

### 2. Installa le dipendenze

```bash
pip install streamlit pypdf2 datapizza-ai
```

### 3. Configura le API Keys

Puoi inserire la tua **OpenAI API key** direttamente nell'interfaccia Streamlit (sidebar) oppure creare un file `.env`:

```bash
cp .env.example .env
# Modifica il file .env con il tuo editor preferito
```

## 📂 Struttura del Progetto

```
DATAPIZZA/
├── run.py                 # Interfaccia UI Streamlit
├── rag_logic.py          # Logica di business RAG
├── start.sh              # Script per avviare l'app (macOS/Linux)
├── requirements.txt      # Dipendenze Python
├── sample_document.txt   # Documento di esempio
└── README.md            # Questa guida
```

## 🎯 Utilizzo

### Metodo 1: Script di Avvio (Consigliato per macOS/Linux)

```bash
./start.sh
```

Lo script si occuperà di:
- Attivare l'ambiente virtuale
- Verificare le dipendenze
- Avviare l'applicazione

### Metodo 2: Manuale

```bash
# Attiva l'ambiente virtuale
source env_project/bin/activate

# Avvia l'applicazione
streamlit run run.py
```

L'applicazione si aprirà automaticamente nel tuo browser all'indirizzo `http://localhost:8501`

### Workflow

1. **Configura le impostazioni** nella sidebar:
   - Inserisci la tua OpenAI API key
   - Seleziona il modello LLM (es. `gpt-4o-mini`)
   - Seleziona il modello di embedding (es. `text-embedding-3-small`)
   - Imposta il numero di documenti da recuperare (k)

2. **Carica i documenti**:
   - Clicca su "Browse files" e seleziona uno o più file (PDF o TXT)
   - Clicca su "🚀 Indicizza Documenti"
   - Attendi che l'indicizzazione sia completata

3. **Fai domande**:
   - Usa la chat per fare domande sui tuoi documenti
   - Il sistema recupererà automaticamente i contenuti rilevanti
   - Visualizza le fonti cliccando su "📚 Fonti"

## 🏗️ Architettura

Il sistema utilizza una **DagPipeline** di datapizza.ai con i seguenti moduli:

```
User Query
    ↓
[Rewriter] → Riformula la query per migliorare il retrieval
    ↓
[Embedder] → Genera embedding della query
    ↓
[Retriever] → Recupera i documenti più rilevanti da Qdrant
    ↓
[Prompt] → Costruisce il prompt con query + contesto
    ↓
[Generator] → Genera la risposta finale con OpenAI
    ↓
Response
```

## 🔧 Configurazione Qdrant

### Modalità In-Memory (Default)

Per default, l'applicazione usa Qdrant in modalità in-memory. Non è necessario installare nulla, ma i dati vengono persi alla chiusura dell'app.

### Modalità Server (Persistente)

Per usare Qdrant con persistenza:

1. **Installa Qdrant** con Docker:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

2. **Disabilita l'opzione "Usa Qdrant in-memory"** nella sidebar
3. **Inserisci host e porta** (default: localhost:6333)

## 📦 Dipendenze Principali

- `streamlit` - Framework per l'interfaccia web
- `datapizza-ai` - Framework RAG
  - `datapizza-ai-core` - Moduli core
  - `datapizza-ai-clients-openai` - Client OpenAI
  - `datapizza-ai-embedders-openai` - Embedder OpenAI
  - `datapizza-ai-vectorstores-qdrant` - Integration Qdrant
- `pypdf2` - Parsing PDF
- `qdrant-client` - Client Qdrant
- `openai` - SDK OpenAI

## 🎨 Personalizzazione

### Modificare il chunk size

Nel file `rag_app.py`, modifica la funzione `chunk_text`:

```python
def chunk_text(text, chunk_size=500, overlap=50):  # Modifica questi valori
    ...
```

### Modificare il prompt template

Modifica i template nel modulo `ChatPromptTemplate`:

```python
ChatPromptTemplate(
    user_prompt_template="Il tuo template custom: {{user_prompt}}",
    retrieval_prompt_template="{% for chunk in chunks %}{{ chunk.text }}{% endfor %}"
)
```

### Aggiungere altri formati di file

Estendi la funzione di upload e aggiungi parser per altri formati (DOCX, CSV, etc.)

## 🐛 Troubleshooting

### Errore: "No matching distribution found for datapizza-ai"

Verifica di avere Python >= 3.10:

```bash
python --version
```

### Errore: "Connection refused" (Qdrant)

Se usi la modalità server, assicurati che Qdrant sia in esecuzione:

```bash
docker ps  # Verifica che il container sia attivo
```

### Errore: "Rate limit exceeded" (OpenAI)

Stai superando i limiti della tua API key. Attendi qualche minuto o passa a un tier superiore.

## 📝 Requisiti

- Python >= 3.10
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))
- (Opzionale) Docker per Qdrant persistente

## 📄 Licenza

Questo progetto è fornito as-is per scopi educativi.

## 🤝 Contributi

Sentiti libero di aprire issue o pull request per miglioramenti!

---

**Made with ❤️ using datapizza.ai 🍕**

