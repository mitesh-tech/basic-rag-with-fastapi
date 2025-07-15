from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import requests
import uuid

# --------- Configuration ---------
GEMINI_API_KEY = "AIzaSyA0lcmW-8KLp3u5k0YPlBk12qnpLVCDtZo"
QDRANT_COLLECTION = "rag_docs"

# --------- FastAPI Init ---------
app = FastAPI()

# --------- Models ---------
class QueryRequest(BaseModel):
    query: str

# --------- Global Objects ---------
qdrant = QdrantClient(host="localhost", port=6333)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------- Startup: Init Collection and Load Docs ---------
@app.on_event("startup")
def startup():
    print("Starting up and initializing Qdrant collection...")

    # Recreate collection if not exists
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Example documents
    documents = [
        "Artificial Intelligence is the simulation of human intelligence processes by machines.",
        "RAG stands for Retrieval-Augmented Generation. It combines search with LLMs.",
        "Qdrant is a vector similarity search engine that stores embeddings.",
        "I live in Pune India.I like to roam around"
    ]

    points = [
        PointStruct(
            id=uuid.uuid4().int >> 64,
            vector=embed_model.encode(doc).tolist(),
            payload={"text": doc}
        )
        for doc in documents
    ]

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    print("Documents inserted into Qdrant.")

# --------- Helper: Retrieve Context ---------
def retrieve_context(query: str, top_k=2):
    query_vector = embed_model.encode(query).tolist()
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )
    return "\n".join([hit.payload['text'] for hit in hits])

# --------- Helper: Call Gemini API ---------
def call_gemini_api(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {response.text}")

# --------- RAG Endpoint ---------
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query
    context = retrieve_context(query)

    prompt = f"""Use the following context to answer the question:
Context:
{context}

Question: {query}
"""
    answer = call_gemini_api(prompt)
    return {"answer": answer}
