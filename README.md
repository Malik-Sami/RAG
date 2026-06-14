# Simple RAG Chatbot

The most minimal Retrieval-Augmented Generation (RAG) chatbot possible.
One Python file. One text document. Zero frameworks.

## What it does

Lets you chat with a document using natural language. It finds the most
relevant passages from the file and feeds them to a local LLM to answer
your question — no hallucination, no internet required.

## Setup

**1. Install Ollama** from https://ollama.ai, then pull the two models:
```bash
ollama pull gemma3
ollama pull nomic-embed-text
```

**2. Install the Python dependency:**
```bash
pip install ollama
# or with uv:
uv add ollama
```

**3. Run it:**
```bash
python simple_rag.py
```

## Try asking

- What is dynamic typing?
- How do I handle errors in Python?
- What is the difference between a list and a dictionary?
- How do virtual environments work?

## Files

```
simple_rag/
├── simple_rag.py              # The entire RAG pipeline (~100 lines)
├── python_fundamentals.txt    # The knowledge base
└── requirements.txt
```
## Update: Persistent Vector Storage

- Replaced temporary in-memory embeddings with ChromaDB
- Added persistent local vector database (`chroma_db/`)
- Eliminated redundant re-embedding on every startup
- Switched from manual cosine similarity to Chroma’s built-in vector search
