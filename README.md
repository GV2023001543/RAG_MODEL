# TeamDino RAG Workspace

TeamDino is a Streamlit study workspace for asking grounded questions across a
set of uploaded documents. It supports PDF, Word, PowerPoint, Excel, CSV, text,
and Markdown files.

## What it includes

- Multiple documents per conversation, with safe replacement and removal
- Hybrid semantic and keyword retrieval
- File and PDF page citations with expandable source passages
- Persistent conversations, documents, and indexes rebuilt after a restart
- Per-browser conversation history backup and automatic restoration
- Optional OCR for scanned PDF pages
- Optional Tavily web search, stock lookup, and calculator tools
- Encrypted personal and system API keys with masked UI display
- Concurrent round-robin rotation, health cooldowns, and provider fallback
- Light and dark themes, conversation search, prompt starters, and text export

## Run locally

Use Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run langgraph_rag_frontend.py
```

Set `RAG_KEY_ENCRYPTION_KEY` to a long random deployment secret before saving
keys. If it is omitted, the app creates a server-local key in `.rag_data/` for
development. Changing or losing this secret makes encrypted keys unreadable.

Users can add personal OpenAI, Groq, Gemini, Anthropic, OpenRouter, or compatible
provider keys in **Settings > Personal AI keys**. Personal keys are always tried
before the system pool. Only masked values are returned to the interface.

Set `ADMIN_PASSWORD` to enable **Settings > Admin pool**. From there, an admin
can configure endpoints, models, token limits, timeouts, enabled status, and
fallback priority, and can add multiple encrypted keys per provider. Lower
priority numbers are attempted first.

System keys can also be supplied without the UI using comma-separated variables:

```text
OPENAI_API_KEYS=key-1,key-2
GROQ_API_KEYS=key-1,key-2,key-3
GEMINI_API_KEYS=key-1,key-2
ANTHROPIC_API_KEYS=key-1
OPENROUTER_API_KEYS=key-1,key-2
```

`RAG_SYSTEM_KEYS_JSON` is available when JSON configuration is more convenient.
Environment keys are never copied into the database. `TAVILY_API_KEY` remains
optional for web search.

Rate-limited keys cool down for 90 seconds, exhausted-quota keys for 15 minutes,
and transient provider failures for 30 seconds by default. Override these with
the corresponding `RAG_*_COOLDOWN_SECONDS` variables.

Scanned PDFs use `GROQ_VISION_MODEL`. OCR is limited to 12 scanned pages by
default; change `PDF_VISION_OCR_MAX_PAGES` when your account and deployment can
support more. Each file is limited to 25 MB, and each conversation can hold 20
files.

## Test

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Test databases are isolated from local workspace data and provider keys.

## Storage

Local conversations and extracted document chunks are stored in `.rag_data/`,
which is excluded from Git. Uploaded file bytes are not retained after text is
extracted. Delete a conversation in the interface to remove its stored messages
and document chunks.

Conversation titles, messages, and citation metadata are also mirrored to the
current browser's `localStorage`. This lets the same browser restore chat history
if server-side conversation rows are lost. API keys and extracted document text
are excluded from that browser snapshot; documents must be uploaded again if the
server-side index is unavailable. Clearing browser site data removes the backup.

## Keep the deployed app awake

The workflow in `.github/workflows/streamlit-keepalive.yml` wakes the app every
four hours and verifies `/_stcore/health`. Configure it once:

1. Open **Settings > Secrets and variables > Actions > Variables** in GitHub.
2. Set `STREAMLIT_APP_URL` to the canonical URL, such as
   `https://your-app-name.streamlit.app` (no query string).
3. Open **Actions > Streamlit Keepalive** and run it manually once.

The manual run also accepts a temporary URL override for troubleshooting. A
missing or unhealthy URL fails visibly instead of recording a false success.

## Main files

- `langgraph_rag_frontend.py` — Streamlit interface
- `langgraph_rag_backend.py` — retrieval and LangGraph workflow
- `rag_ingestion.py` — document extraction and OCR
- `rag_providers.py` — model and search provider configuration
- `provider_adapters.py` — common provider adapter registry
- `key_vault.py` — encryption, key health, and round-robin scheduling
- `rag_storage.py` — persistent SQLite storage
- `browser_history.py` — validated browser-local history snapshots
- `assets/workspace.css` — responsive visual design
