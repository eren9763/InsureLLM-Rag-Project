# InsureLLM Assistant

A RAG-based chatbot built for a fictional insurance company knowledge base.  
It reads internal markdown documents, turns them into searchable chunks, stores embeddings in ChromaDB, and answers questions through a Gradio interface with source attribution and response evaluation.

<img width="1894" height="564" alt="image" src="https://github.com/user-attachments/assets/f1495b77-cc33-42df-84e8-eb4ac037418b" />


## Overview

This project is a small end-to-end Retrieval-Augmented Generation system designed to answer questions about **InsureLLM**, a fictional company with internal documents such as employee profiles, products, and contracts.

Instead of relying only on the model’s built-in knowledge, the app retrieves relevant chunks from a local knowledge base and uses them as context for generation. The interface also shows source files used in the answer and displays basic evaluation signals like faithfulness and answer relevancy.

## Features

- Retrieval-Augmented Generation pipeline over local markdown files
- LLM-based document chunking with overlap
- Embeddings generated with `BAAI/bge-m3`
- Local vector storage with ChromaDB
- Gradio chat interface for interactive Q&A
- Source attribution shown under responses
- Response evaluation with:
  - Faithfulness
  - Answer relevancy
  - Short explanation of the evaluation result

## How it works

The pipeline follows a simple RAG workflow:

1. Documents are loaded from the `knowledge-base/` directory.
2. Each markdown file is split into overlapping chunks.
3. Chunk embeddings are created with HuggingFace embeddings (`BAAI/bge-m3`).
4. Embeddings and metadata are stored in a local ChromaDB collection.
5. When a user asks a question, the app retrieves the most relevant chunks.
6. The retrieved context is passed to the language model to generate an answer.
7. The answer is evaluated for faithfulness and relevancy, and the UI shows the results.

## Tech stack

- **Python**
- **LiteLLM** for model calls
- **OpenAI-compatible API** for chunking, answering, and evaluation
- **HuggingFace Embeddings** (`BAAI/bge-m3`)
- **ChromaDB** for vector storage
- **Gradio** for the user interface
- **Pydantic** for structured outputs and validation

## Project structure

```text
.
├── rag_project.py          # Main application logic
├── knowledge-base/         # Markdown documents used as the knowledge source
├── preprocessed_db/        # Local ChromaDB persistence directory
├── .env.example            # Example environment variables
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your API key:

```bash
copy .env.example .env
```

Then set:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run the app

```bash
python rag_project.py
```

On the first run, the application processes the markdown files in `knowledge-base/` and builds a local vector database in `preprocessed_db/`.

On later runs, it reuses the existing vector store instead of rebuilding everything from scratch.

## Example use cases

You can ask questions like:

- Who is Avery Lancaster?
- What insurance products does InsureLLM offer?
- What information appears in the company contracts?
- Which employees are mentioned in the knowledge base?

## Why this project is interesting

This project goes beyond a minimal RAG demo by including a few practical features that make the system easier to inspect and trust:

- retrieved context is used for grounded answers
- source files are shown in the interface
- generated answers are evaluated after generation
- the vector store is persisted locally for faster reuse

## Limitations

- The knowledge base is small and local
- Retrieval is currently based on vector similarity only
- Evaluation is lightweight and not a replacement for a full benchmark
- The system depends on the quality of chunking and document formatting

## Possible improvements

Some natural next steps for the project would be:

- hybrid search (dense + keyword retrieval)
- reranking retrieved chunks
- automatic abstention when context is insufficient
- batch evaluation on a fixed QA set
- better logging and observability
- a more polished Gradio interface

## Requirements

- Python 3.11+
- An API key for the model provider you use through LiteLLM

## Notes

`preprocessed_db/` is generated locally and should usually be added to `.gitignore`.

If you update the documents in `knowledge-base/`, you may want to delete the existing vector store and rebuild it so the retrieval index stays in sync.
