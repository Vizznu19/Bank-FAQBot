import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from gtts import gTTS
import os
import tempfile

# Create the FastAPI app
app = FastAPI()

# Allow CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for deployment
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load data
faq_df = pd.read_csv("BankFAQs.csv", usecols=["Question", "Answer"])
questions = faq_df["Question"].astype(str).tolist()
answers = faq_df["Answer"].astype(str).tolist()

# Chunking function: split text into sentences
sentence_splitter = re.compile(r'(?<=[.!?]) +')
def chunk_text(text):
    return [chunk.strip() for chunk in sentence_splitter.split(text) if chunk.strip()]

# Prepare chunked data
chunked_questions = []  # Parent question for each chunk
chunks = []             # The actual chunk text
chunked_answers = []    # Full answer for reference
for q, a in zip(questions, answers):
    answer_chunks = chunk_text(a)
    for chunk in answer_chunks:
        chunked_questions.append(q)
        chunks.append(chunk)
        chunked_answers.append(a)

# Load model and build index
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")
chunk_index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
chunk_index.add(chunk_embeddings)

class QueryRequest(BaseModel):
    query: str
    k: int = 1

class TTSRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/search")
async def search_faq(req: QueryRequest):
    query_embedding = model.encode([req.query]).astype("float32")
    D, I = chunk_index.search(query_embedding, req.k)
    # Calculate cosine similarity from L2 distance
    # cosine_sim = 1 - (L2_distance^2 / 2)
    similarities = 1 - (D[0] / 2)
    threshold = 0.6
    results = []
    for idx, sim in zip(I[0], similarities):
        if sim >= threshold:
            results.append({
                "question": chunked_questions[idx],
                "full_answer": chunked_answers[idx]
            })
    return {"results": results}

@app.post("/tts")
async def text_to_speech(req: TTSRequest):
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        # Convert text to speech
        tts = gTTS(text=req.text, lang='en', slow=False)
        tts.save(tmp_file.name)
        
        # Return the audio file
        return FileResponse(tmp_file.name, media_type="audio/mpeg", filename="speech.mp3")

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "FAQ Assistant is ready"} 