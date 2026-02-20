# 📜 Smart Contract Assistant

A RAG-based contract analysis chatbot powered by NVIDIA AI endpoints, LangChain, and Gradio.

## Features

- Upload PDF or DOCX documents
- Ask questions about the document content
- Conversation memory across turns
- Guardrail filtering for off-topic questions

## Requirements

Install dependencies:

```bash
pip install langchain langchain-community langchain-nvidia-ai-endpoints langchain-chroma langserve[all] fastapi uvicorn gradio python-dotenv pypdf docx2txt
```

## Setup

Create a `.env` file in the project root with your NVIDIA API key:

```
NVIDIA_API_KEY=your_api_key_here
```

## How to Run

The app has two separate servers that must both be running.

**1. Start the backend (RAG server) — in one terminal:**

```bash
python server.py
```

Runs on `http://localhost:9013`

**2. Start the frontend (Gradio UI) — in another terminal:**

```bash
python app.py
```

Runs on `http://localhost:9012`

**3. Open your browser and go to:**

```
http://localhost:9012
```

## Usage

1. Upload a PDF or DOCX file using the left panel
2. Wait for the ✅ confirmation message
3. Start chatting with the document in the chat panel
