# Contexta

Contexta is a Streamlit app for searching uploaded study documents, asking
grounded questions about them, and grading student answers against retrieved
reference material.

## Features

- Upload PDF, TXT, and DOCX study materials.
- Index documents locally and search relevant passages.
- Ask questions grounded in the indexed materials.
- Grade answers against a retrieved reference passage, with a selectable
  maximum of 1–10 marks.
- Use an OpenAI-compatible LLM API, configured here for Groq.

## Setup

Use Python 3.11 or newer. From the project directory, create and activate a
virtual environment, then install the dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `LLM_API_KEY` to your provider API key.
The default Groq configuration is:

```dotenv
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
```

Never commit `.env` or share API keys. The repository `.gitignore` excludes
local environment files, uploaded documents, local databases, and Python
environments.

## Run the app

```powershell
python -m streamlit run frontend/app.py
```

Upload study materials from the app to index them. Documents and the ChromaDB
index are stored locally under `data/` and are intentionally not included in
the GitHub project.

## Optional API backend

The repository also includes a small FastAPI health-check service:

```powershell
uvicorn backend.main:app --reload
```
