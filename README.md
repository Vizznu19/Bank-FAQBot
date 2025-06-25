# Bank FAQ Assistant

A vector-based FAQ assistant for bank queries using sentence transformers and FAISS for semantic search.

## Features

- **Semantic Search**: Uses sentence-transformers (all-MiniLM-L6-v2) for understanding question meaning
- **Chunking**: Splits long answers into sentences for more precise matching
- **Similarity Threshold**: Only returns answers above a certain relevance threshold
- **Modern UI**: Clean, gradient-styled interface

## How to Use

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Start the backend:
   ```
   uvicorn main:app --reload
   ```

3. Open `static/index.html` in your browser

4. Type your question and get instant answers!

## Technical Details

- **Backend**: FastAPI with sentence-transformers and FAISS
- **Frontend**: HTML/CSS/JavaScript with modern gradient design
- **Data**: Bank FAQ dataset with questions and answers
- **Search**: Semantic similarity with chunking for precision

## API Endpoints

- `GET /`: Health check
- `POST /search`: Search for FAQ answers
  - Body: `{"query": "your question", "k": 1}`
  - Returns: `{"results": [{"question": "...", "full_answer": "..."}]}`

---

*Built with FastAPI, sentence-transformers, and FAISS for efficient semantic search.* 