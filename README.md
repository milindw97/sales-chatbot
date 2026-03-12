# Sales Call Chatbot 🤖

A conversational AI chatbot that helps sales users understand and summarize their past sales calls using RAG (Retrieval-Augmented Generation).

## 📋 Overview

This chatbot:
- Ingests sales call transcripts and stores them in a vector database (FAISS)
- Allows natural language queries about call content
- **Conversation memory**: Remembers context for follow-up questions
- Provides answers with **source attribution** (call ID + timestamps)
- Supports summarization of individual calls
- Configurable LLM providers (Ollama, Gemini)
- Configurable embedding providers (Sentence Transformers, Gemini)

## 🏗️ Architecture

```
sales-chatbot/
├── src/
│   ├── config.py              # Configuration management
│   ├── ingestion/
│   │   ├── parser.py          # Transcript parsing & chunking
│   │   └── service.py         # Ingestion orchestration
│   ├── storage/
│   │   ├── database.py        # SQLite metadata storage
│   │   ├── embeddings.py      # Embedding providers
│   │   └── vector_store.py    # FAISS vector store wrapper
│   ├── retrieval/
│   │   └── service.py         # RAG implementation
│   └── llm/
│       ├── providers.py       # LLM providers (Ollama/Gemini)
│       └── prompts.py         # Centralised prompt templates
├── tests/                     # pytest test suite (131 tests)
│   ├── conftest.py            # Shared fixtures
│   ├── test_parser.py
│   ├── test_database.py
│   ├── test_vector_store.py
│   ├── test_ingestion_service.py
│   ├── test_retrieval_service.py
│   └── test_providers.py
├── data/
│   ├── transcripts/           # Raw call transcripts
│   └── db/                    # SQLite + FAISS index
├── cli.py                     # Main CLI interface
├── requirements.txt           # Dependencies
└── .env                       # Configuration (create from .env.example)
```

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | FAISS | Fast similarity search |
| **Metadata DB** | SQLite + SQLAlchemy | Store call metadata & chunks |
| **Embeddings** | Sentence Transformers / Gemini | Convert text to vectors |
| **LLM** | Ollama (llama3.2) / Gemini | Generate answers |
| **Chunking** | Custom parser | Split transcripts with overlap |

## 🚀 Setup

### Prerequisites

- Python 3.12+
- (Optional) Ollama installed locally
- (Optional) Gemini API key

### Installation

1. **Clone/Download the project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings
```

### Configuration Options

Edit `.env` file:

```bash
# Choose LLM: "ollama" or "gemini"
LLM_PROVIDER=ollama

# If using Gemini, add your API key
GEMINI_API_KEY=your_key_here

# Choose embeddings: "sentence-transformers" or "gemini"
EMBEDDING_PROVIDER=sentence-transformers

# Ollama settings (if using Ollama)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Features
MAX_HISTORY_TURNS=5
```

### For Ollama Users

1. Install Ollama: https://ollama.ai
2. Pull llama3.2: `ollama pull llama3.2`
3. Ensure Ollama is running: `ollama serve`

### For Gemini Users

1. Get API key: https://makersuite.google.com/app/apikey
2. Set in `.env`: `GEMINI_API_KEY=your_key_here`

## 📥 Loading Sample Data

Copy the provided transcript files to `data/transcripts/`:

```bash
cp /path/to/transcripts/*.txt data/transcripts/
```

Then run the chatbot and ingest them:

```bash
python cli.py
# At the prompt:
> ingest all
```

## 💬 Usage

### Start the Chatbot

```bash
python cli.py
```

### Available Commands

#### 📋 List Calls
```
> list calls
> list my call ids
```

#### 📥 Ingest Transcripts
```
> ingest all
> ingest data/transcripts/1_demo_call.txt
```

#### 📝 Summarize Calls
```
> summarize demo_call
> summarize the last call
```

#### ❓ Ask Questions
Just type naturally! Examples:

```
> What objections were raised in the demo call?
> Give me all negative comments when pricing was mentioned
> What were the main security concerns?
> What next steps were agreed upon in the negotiation call?
> Who are the key participants across all calls?
```

#### 🔧 Utility
```
> reset      # Clear conversation history
> help       # Show help
> clear      # Clear screen
> exit       # Quit
```

## 📊 Example Session

```
You > ingest all

📄 Processing: data/transcripts/1_demo_call.txt
✓ Created transcript record: demo_call
  Participants: Jordan, Luis, Priya
  Duration: 06:37
  Total chunks: 15
✅ Successfully ingested 'demo_call' with 15 chunks

You > What were the main objections raised?

💡 ANSWER:
Based on the call transcripts, several key objections were raised:

1. Integration concerns (demo_call [02:59]):
   - Need for Okta SSO
   - Concerns about sandbox support

2. Pricing comparisons (pricing_call [01:24]):
   - Competitor X offering lower price with "unlimited minutes"
   
3. Security requirements (objection_call [03:35]):
   - PII redaction needs
   - Data residency requirements

📚 SOURCES (3):
1. Call: demo_call
   Time: [02:54] - [03:20]
   Relevance: 0.892
   Snippet: SE: Let's pop open the Copilot chat...

You > Which integrations specifically?

🔄 (Searched for: 'Which specific integrations were mentioned causing concerns?')
💡 ANSWER:
======================================================================
The main integration discussed was Okta SSO. Priya (RevOps) brought this up 
because their IT mandate requires Okta for all new vendor deployments...

📚 SOURCES (1):
1. Call: demo_call
   Time: [02:54] - [03:20]
   Relevance: 0.910
   Snippet: Priya: Does this support Okta SSO? IT won't approve without it.

You > summarize demo_call

📊 SUMMARY: demo_call
======================================================================
This was a product demo call between Jordan (AE), Luis (SE) and 
Priya (RevOps Director). The main purpose was to demonstrate the 
AI Copilot features for call analysis.

Key Discussion Points:
- Dashboard showing call health scores and adoption metrics
- Coaching features for managers
- Integration requirements (Okta SSO, sandbox)
- Multilingual support (Hindi-English)
- Pricing: ₹1,800/user/month with 20% pilot discount
...
```

## 🎯 Design Decisions

### Storage Schema

**SQLite Tables:**
- `call_transcripts`: Call metadata (call_id, filename, ingestion_date)
- `call_chunks`: Individual chunks (text, timestamp_range, faiss_index)

**FAISS Index:**
- Stores dense vector embeddings
- Enables fast similarity search (L2 distance)
- Index position mapped to chunk via `faiss_index` field

### Why This Architecture?

1. **Modularity**: Easy to swap LLM/embedding providers
2. **Scalability**: FAISS handles millions of vectors efficiently
3. **Source Attribution**: SQLite maintains chunk→call mapping
4. **Flexibility**: Support multiple providers without code changes

### Chunking Strategy

- **Size**: 512 characters per chunk
- **Overlap**: 50 characters
- **Benefit**: Maintains context across chunk boundaries
- **Trade-off**: Slightly redundant but better recall

### RAG Implementation

1. **Query** → Embed with same provider as documents
2. **Search** → FAISS returns top-k similar chunks
3. **Retrieve** → Map FAISS indices to SQLite chunks
4. **Augment** → Build context with [Source X] markers
5. **Generate** → LLM produces answer citing sources

## 🧪 Testing

The project has a comprehensive pytest test suite covering all core modules:

```bash
# Run all tests
pytest tests/ -q

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run a specific module
pytest tests/test_parser.py -v
```

**131 tests · 87% coverage** across:
- `tests/test_parser.py` — TranscriptParser, chunking edge cases, metadata extraction
- `tests/test_database.py` — All DatabaseManager methods, DetachedInstanceError regression
- `tests/test_vector_store.py` — FAISS add/search/save/load, error cases
- `tests/test_ingestion_service.py` — End-to-end ingestion, duplicate handling, directory scanning
- `tests/test_retrieval_service.py` — RAG query, summarization, LLM error fallback
- `tests/test_providers.py` — Factory functions, provider unit tests (no real API calls)

## 🔄 Extending the System

### Add New LLM Provider

1. Create provider class in `src/llm/providers.py`
2. Implement `LLMProvider` interface
3. Add to `get_llm_provider()` factory
4. Add any new system prompts to `src/llm/prompts.py`
5. Update `.env.example`

### Add New Embedding Provider

1. Create provider class in `src/storage/embeddings.py`
2. Implement `EmbeddingProvider` interface
3. Add to `get_embedding_provider()` factory

### Add New Features

Ideas for live coding extension:
- Export summaries to PDF/DOCX
- Advanced filters (by participant, date range)
- Sentiment analysis on chunks
- Automatic action item extraction

## 📝 Assumptions

1. **File Format**: Transcripts are plain text with `[MM:SS]` timestamps
2. **Language**: Primarily English (multilingual supported in transcripts)
3. **Call IDs**: Derived from filename (e.g., `1_demo_call.txt` → `demo_call`)
4. **Single User**: No multi-tenancy (could be added with user_id field)
5. **Local Execution**: Designed for local/single-machine deployment

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error: Failed to connect to Ollama
```
**Fix**: Ensure Ollama is running: `ollama serve`

### Import Error
```
ModuleNotFoundError: No module named 'sentence_transformers'
```
**Fix**: Install dependencies: `pip install -r requirements.txt`

### Empty Results
```
I couldn't find any relevant information
```
**Fix**: Check if calls are ingested: `list calls`

### FAISS Dimension Mismatch
```
Vector dimension doesn't match index
```
**Fix**: Delete `data/db/` and re-ingest with same embedding provider

## 📚 References

- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/
- **Ollama**: https://ollama.ai/
- **Gemini API**: https://ai.google.dev/

## 👤 Author

Built for I2 Coding Round - Conversational AI Copilot Assignment

## 📄 License

For assignment evaluation purposes only.
