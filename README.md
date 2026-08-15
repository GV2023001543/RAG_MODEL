# TeamDino RAG Model

A Streamlit chatbot for college notes and uploaded documents, powered by
LangGraph, FAISS, HuggingFace embeddings, Tavily search, yfinance, and Groq.

## Features

- Chat with the built-in college knowledge base in `college_data/`
- Upload documents for RAG over PDF, Word, PowerPoint, Excel, CSV, text, and Markdown
- Use vision OCR for scanned PDFs when a Groq API key is available
- Generate cheat sheets, flashcards, summaries, concept explanations, and diagram explanations
- Use web search, stock lookup, and calculator tools from the chat flow
- Export conversations as text

## Project Files

- `langgraph_rag_frontend.py` - Streamlit UI
- `langgraph_rag_backend.py` - LangGraph workflow, tools, ingestion, and RAG logic
- `college_rag.py` - Shared index builder for `college_data/`
- `college_data/` - Markdown knowledge base files
- `.env.example` - API key template

## Setup

1. Install Python 3.11.
2. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Create `.env` from `.env.example` and add your keys:

```text
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

5. Run the app:

```powershell
python -m streamlit run langgraph_rag_frontend.py
```

## Keep Streamlit Cloud Awake

Streamlit Community Cloud can hibernate apps after inactivity. This repo includes
a GitHub Actions workflow at `.github/workflows/keep-streamlit-awake.yml` that
pings the deployed app every 6 hours.

After deploying, add this GitHub repository variable:

```text
STREAMLIT_APP_URL=https://your-app-name.streamlit.app
```

Then open the workflow named `Keep Streamlit App Awake` and run it once manually
to confirm the URL is correct.
