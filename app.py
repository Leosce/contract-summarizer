import gradio as gr
from rag_logic import process_uploaded_file, guardrail_chain, get_rag_chain

# Global state to keep the retriever in memory
current_retriever = None

def handle_upload(file):
    global current_retriever
    if file:
        try:
            current_retriever = process_uploaded_file(file.name)
            return "✅ Document processed! You can now ask questions."
        except Exception as e:
            return f"❌ Error processing file: {e}"
    return "❌ Upload failed."

def chat_func(message, history):
    global current_retriever
    if not current_retriever:
        return "Please upload a document (PDF/DOCX) first!"
    
    # Run Guardrail
    guard_result = guardrail_chain.invoke({"question": message})
    
    # ADD THIS CHECK: Handle None or unexpected return types
    if guard_result is None:
        return "⚠️ Error: The guardrail could not process this request. Please try again."
        
    if not guard_result.is_relevant:
        return f"⚠️ Guardrail: {guard_result.reasoning}"
    
    # Run RAG
    chain = get_rag_chain(current_retriever)
    return chain.invoke(message)

# Removed theme from constructor
with gr.Blocks() as demo:
    gr.Markdown("# 📜 Smart Contract Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload Contract", file_types=[".pdf", ".docx"])
            status = gr.Textbox(label="Status", interactive=False)
            file_upload.change(handle_upload, inputs=[file_upload], outputs=[status])
            
        with gr.Column(scale=3):
            # Removed 'type="messages"' to fix TypeError
            gr.ChatInterface(fn=chat_func)

if __name__ == "__main__":
    # Moved theme here
    demo.launch(theme=gr.themes.Soft())