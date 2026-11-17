
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import MasonBroRAG


rag_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_agent

    print("Starting MasonBro AI Agent...")

    rag_agent = MasonBroRAG()

    # Index FAQ knowledge base
    rag_agent.index_knowledge()

    print("MasonBro AI Agent is ready!")

    yield

    print("MasonBro AI Agent stopped.")


app = FastAPI(
    title="MasonBro AI Agent",
    description="RAG-powered MasonBro customer support API",
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://masonbro.com",
        "https://www.masonbro.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


# Request model
class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=2000
    )


# Home endpoint
@app.get("/")
def home():

    return {
        "message": "MasonBro AI Agent is running!",
        "status": "success"
    }


# Health endpoint
@app.get("/health")
def health():

    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="AI Agent is not initialized"
        )

    return {
        "status": "healthy",
        "knowledge_chunks": rag_agent.collection.count()
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="AI Agent is not ready"
        )

    try:

        result = rag_agent.ask(request.message)

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as error:

        print("Chat error:", str(error))

        raise HTTPException(
            status_code=500,
            detail="Unable to generate a response"
        )
