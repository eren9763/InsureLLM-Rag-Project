# InsureLLM Rag Project

A retrieval-augmented generation (RAG) chatbot for the fictional company Insurellm. The app chunks markdown documents from the knowledge base, embeds them with HuggingFace (`BAAI/bge-m3`), stores vectors in ChromaDB, and answers questions through a Gradio UI.

## Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
copy .env.example .env
```

4. Run the app:

```bash
python rag_project.py
```

On first run, the app processes documents in `knowledge-base/` and builds a local ChromaDB store in `preprocessed_db/`. Later runs reuse the existing vector store.

## Project structure

- `rag_project.py` — main RAG pipeline and Gradio UI
- `knowledge-base/` — markdown documents (employees, products, contracts)
- `preprocessed_db/` — generated vector store (ignored by git)

## Requirements

- Python 3.11+
- OpenAI API key (used via LiteLLM for chunking, chat, and evaluation)
