from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import fitz
import docx
import pickle
import faiss
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -------- Load environment variables --------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -------- Initialize FastAPI --------
app = FastAPI(title="AI Document Q&A Assistant")

# -------- Secure CORS Setup --------
allowed_origins = [
    "https://your-frontend-domain.com",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# -------- Config --------
UPLOAD_FOLDER = "app/documents"
VECTOR_STORE_PATH = "app/vector_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]

# -------- Global Memory Cache --------
vector_indices = {}
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- Helpers --------


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(file_path: str) -> str:
    ext = file_path.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif ext == "docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


def save_index(index, chunks, filename):
    faiss.write_index(index, os.path.join(
        VECTOR_STORE_PATH, f"{filename}.index"))
    with open(os.path.join(VECTOR_STORE_PATH, f"{filename}_chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)


def load_index(filename):
    index_path = os.path.join(VECTOR_STORE_PATH, f"{filename}.index")
    chunks_path = os.path.join(VECTOR_STORE_PATH, f"{filename}_chunks.pkl")
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise HTTPException(
            status_code=404, detail="FAISS index not found for this document.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# -------- Auto-load all FAISS indexes at startup --------


@app.on_event("startup")
def load_all_indexes():
    print("🔄 Loading FAISS indexes...")
    for file in os.listdir(VECTOR_STORE_PATH):
        if file.endswith(".index"):
            name = file.replace(".index", "")
            try:
                idx, chunks = load_index(name)
                vector_indices[name] = (idx, chunks)
                print(f"✅ Loaded index for {name}")
            except Exception as e:
                print(f"⚠️ Could not load {name}: {e}")
    print(f"Total indexes loaded: {len(vector_indices)}")

# -------- Upload Endpoint --------


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text(file_path)
    chunks = chunk_text(text)
    embeddings = embedding_model.encode(
        chunks, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_index(index, chunks, file.filename)
    vector_indices[file.filename] = (index, chunks)

    return {"message": f"✅ Uploaded and indexed {file.filename}", "chunks": len(chunks)}

# -------- Query Model --------


class QueryRequest(BaseModel):
    filename: str
    question: str
    top_k: int = 3


def generate_answer(question, context):
    prompt = f"""
You are a financial document assistant.
Use the provided context to answer.
If not found, say "Not found in the document."

Context:
{context}

Question: {question}
Answer:
"""
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": "meta-llama/llama-3.1-8b-instruct",
              "messages": [{"role": "user", "content": prompt}]},
    )
    data = response.json()
    if "choices" not in data:
        raise HTTPException(status_code=500, detail="LLM failed to respond.")
    return data["choices"][0]["message"]["content"].strip()

# -------- Query Endpoint --------


@app.post("/query")
async def query(req: QueryRequest):
    if req.filename not in vector_indices:
        raise HTTPException(
            status_code=404, detail="File not indexed. Upload first.")

    index, chunks = vector_indices[req.filename]
    query_embedding = embedding_model.encode(
        [req.question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, req.top_k)

    retrieved = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved)
    answer = generate_answer(req.question, context)

    return {"answer": answer, "matches": retrieved[:req.top_k]}
