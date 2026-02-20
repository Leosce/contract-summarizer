import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from pathlib import Path
from typing import List, Any

from rag_logic import (
    process_uploaded_file,
    get_rag_chain,
    guardrail_chain,
    llm,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
current_retriever = None

class QuestionInput(BaseModel):
    question: str = "What is this contract about?"
    history: List[Any] = []

app = FastAPI(
    title="Contract Assistant API",
    version="1.0",
    description="RAG-based contract analysis server",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global current_retriever
    
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {suffix}. Please upload PDF or DOCX."
        )
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        current_retriever = process_uploaded_file(temp_path)
        return {"status": "success", "message": f"{file.filename} processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        Path(temp_path).unlink(missing_ok=True)

add_routes(
    app,
    guardrail_chain,
    path="/guardrail",
)

add_routes(
    app,
    RunnableLambda(
        lambda x: get_rag_chain(current_retriever).invoke({
            "question": x["question"],
            "history": [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in x.get("history", [])
            ]
        })
        if current_retriever
        else "No document uploaded yet."
    ).with_types(input_type=QuestionInput),
    path="/generator",
)
add_routes(app, llm | StrOutputParser(), path="/basic_chat")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9013)