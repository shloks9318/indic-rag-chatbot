# Indic Knowledge RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in the Bhagavad Gita, built to explore RAG architecture over ancient Indic knowledge texts.

## What it does

The system ingests the Bhagavad Gita (TTD Tirumala edition), chunks and embeds the text, stores it in a vector database, and answers user questions by retrieving the most relevant passages and generating a grounded response — refusing to answer when the source text doesn't contain the relevant information.

## Tech Stack

- **LangChain** — orchestration layer connecting retrieval and generation
- **ChromaDB** — vector store for semantic search
- **HuggingFace Sentence Transformers** (`all-MiniLM-L6-v2`) — text embeddings
- **Groq API** (`llama-3.3-70b-versatile`) — LLM inference
- **Streamlit** — chat interface
- **pypdf** — PDF text extraction

## Architecture

```
PDF (Bhagavad Gita)
      ↓
Chunking (1000 chars, 150 overlap)
      ↓
HuggingFace Embeddings → ChromaDB (persisted vector store)
      ↓
User Query → Embedded → Top-8 similar chunks retrieved
      ↓
Chunks + Query → Groq (LLaMA 3.3 70B)
      ↓
Grounded Answer (or explicit "not found in text" response)
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/indic-rag-chatbot.git
cd indic-rag-chatbot
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Run

```bash
streamlit run app.py
```

First run will download the embedding model and index the PDF — takes a few minutes. Subsequent runs load instantly from the persisted vector store.

## Design Notes

- **Strict grounding**: the system is prompted to answer only from retrieved context, and explicitly states when information isn't present in the source text rather than relying on the LLM's general knowledge.
- **Chunk size and `k` tuning**: chunk size of 1000 characters with 150 overlap and retrieval of top-8 chunks were chosen after testing smaller configurations that missed conceptually linked passages split across chunk boundaries.
- **Corpus choice**: the TTD Tirumala edition was selected over alternatives because it includes commentary alongside verses, giving the retriever richer semantic surface area for both direct-verse and conceptual/character queries.

## Known Limitations

- Single-document corpus (no multi-source retrieval yet)
- No conversational memory across turns
- No hybrid (keyword + semantic) search
- No guardrails against prompt injection

## Future Work

- Add conversational memory
- Hybrid search (BM25 + vector) for improved recall on exact terminology
- Multi-document support with metadata filtering
- Source citation (chapter/verse) in responses
