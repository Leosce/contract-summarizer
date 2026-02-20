import gradio as gr
import requests
import uvicorn
from fastapi import FastAPI
from langserve import RemoteRunnable

app = FastAPI(title="Smart Contract Assistant")

# API Endpoints
SERVER_URL = "http://localhost:9013"
rag_chain = RemoteRunnable(f"{SERVER_URL}/generator/")
guardrail_chain = RemoteRunnable(f"{SERVER_URL}/guardrail/")

def handle_upload(file):
    # If the user clears the file, 'file' becomes None
    if file is None:
        return "No document loaded."
        
    try:
        # Simplified: Gradio's 'file' object has a .name attribute (the path)
        with open(file.name, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}/upload",
                files={"file": (file.name, f)}
            )
        
        if response.status_code == 200:
            return f"{response.json().get('message', 'Processed')}"
        return f"Server Error: {response.text}"
        
    except Exception as e:
        return f"Connection Error: {e}"

def get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content)

def chat_func(message, history):
    try:
        guard = guardrail_chain.invoke({"question": message})
        if not getattr(guard, 'is_relevant', True):
            return f"Guardrail: {getattr(guard, 'reasoning', 'Topic not allowed.')}"

        formatted_history = [
            {"role": msg["role"], "content": get_text(msg["content"])}
            for msg in history
            if msg["role"] in ("user", "assistant")
        ]

        result = rag_chain.invoke({
            "question": message,
            "history": formatted_history
        })

        return str(result) if result else "No response generated."

    except Exception as e:
        return f"Error: {str(e)}"
    
# Define the Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("#Smart Contract Assistant")

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload PDF/DOCX", file_types=[".pdf", ".docx"])
            status = gr.Textbox(label="Status", interactive=False)
            
            # This triggers on upload AND on clear
            file_upload.change(handle_upload, inputs=file_upload, outputs=status)

        with gr.Column(scale=3):
            gr.ChatInterface(fn=chat_func)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9012)