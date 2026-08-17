# Agentic RAG

An intelligent Retrieval-Augmented Generation (RAG) system that combines advanced retrieval mechanisms with generative AI to answer questions about company filings and financial documents.

## Features

- 🤖 **Agentic Architecture**: Intelligent agent system that routes queries and orchestrates retrieval and generation
- 📚 **Multi-Document Retrieval**: BM25-based indexing and semantic search across company filings
- 🔄 **Query Reformulation**: Automatic query optimization for better retrieval results
- 📊 **Confidence Scoring**: Reliability assessment of generated responses
- 🌐 **Web Interface**: Modern Next.js frontend for intuitive user interactions
- 🚀 **Fast Performance**: Optimized retrieval and generation pipeline

## Project Structure

```
Agentic_RAG/
├── backend/                    # Python backend services
│   ├── agent.py               # Core agent orchestration
│   ├── main.py                # FastAPI application entry point
│   ├── retriever.py           # Document retrieval logic
│   ├── generator.py           # Text generation module
│   ├── router.py              # Query routing system
│   ├── reformulator.py        # Query reformulation engine
│   ├── confidence.py          # Confidence scoring
│   ├── build_index.py         # Index building utilities
│   └── ingest.py              # Data ingestion pipeline
├── frontend/                   # Next.js web interface
│   ├── app/
│   │   ├── page.tsx          # Main page
│   │   ├── layout.tsx        # Layout component
│   │   ├── globals.css       # Global styles
│   │   └── api/
│   │       └── query/        # API endpoint for queries
│   ├── package.json
│   ├── next.config.ts
│   └── tsconfig.json
└── data/                       # Data files
    ├── chunks.json            # Processed document chunks
    ├── companies.json         # Company metadata
    ├── bm25_index.pkl        # BM25 index
    └── filings/               # Raw company filings
        ├── AAPL.txt
        ├── MSFT.txt
        ├── GOOGL.txt
        └── ... (other companies)
```

## Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI**: Modern web framework
- **LangChain**: LLM orchestration
- **BM25**: Full-text search indexing
- **Pydantic**: Data validation

### Frontend
- **Next.js 14**: React framework
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Styling
- **React**: UI components

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Build the search index:
```bash
python build_index.py
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local with your API endpoint
```

## Usage

### Running the Backend

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### Running the Frontend

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:3000`

## API Documentation

### Query Endpoint

**POST** `/api/query`

Request body:
```json
{
  "query": "What is Apple's revenue for 2023?",
  "company": "AAPL",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "Based on the filing...",
  "confidence": 0.92,
  "sources": [
    {
      "text": "...",
      "score": 0.95
    }
  ]
}
```

## Key Components

### Agent (agent.py)
Orchestrates the RAG pipeline, managing query routing and response generation.

### Retriever (retriever.py)
Handles document retrieval using BM25 indexing and semantic search.

### Generator (generator.py)
Generates coherent answers based on retrieved documents using language models.

### Router (router.py)
Routes queries to appropriate processing pipelines based on query type and content.

### Reformulator (reformulator.py)
Optimizes queries for better retrieval results through reformulation techniques.

### Confidence (confidence.py)
Scores the confidence level of generated responses based on source relevance.

## Configuration

Edit `backend/config.py` or environment variables to customize:
- LLM model selection
- Retrieval parameters (top_k, similarity threshold)
- API port and host
- Database connections

## Performance Optimization

- Implement caching for frequently asked questions
- Use vector databases for semantic search at scale
- Batch process multiple queries
- Implement query result pagination

## Future Enhancements

- [ ] Vector embeddings integration (Pinecone, Weaviate)
- [ ] Multi-language support
- [ ] Real-time document ingestion
- [ ] Advanced analytics dashboard
- [ ] User authentication and history
- [ ] Streaming response support

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Built with FastAPI and Next.js
- Powered by state-of-the-art LLM technologies
- Company filings data from SEC EDGAR
