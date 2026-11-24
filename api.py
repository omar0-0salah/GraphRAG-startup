from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from graph_rag import DATABASE, driver, generate_answer
from retriever import retrieve_context
import uvicorn
import os

class RetrievalRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of companies to retrieve")


app = FastAPI(
    title="Startup GraphRAG Retrieval API",
    description="API for querying the startup knowledge graph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Startup GraphRAG API is running!",
        "docs": "/docs",
        "endpoints": ["/health", "/graph-rag"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/graph-rag")
def graph_rag_endpoint(request: RetrievalRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        retrieval_results = retrieve_context(
            question=question,
            driver=driver,
            database=DATABASE,
            top_k=request.top_k,
        )
        answer = generate_answer(question, retrieval_results)
        return {
            "question": question,
            "answer": answer,
            "retrieval_type": retrieval_results["type"],
            "results": retrieval_results["results"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":


    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
