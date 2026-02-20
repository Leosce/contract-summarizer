# %%
import os
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

# %%
embedder = NVIDIAEmbeddings(
    model="nvidia/nv-embedqa-e5-v5"
)

llm = ChatNVIDIA(
    model="openai/gpt-oss-120b", 
    temperature=0.1,
    max_tokens=1024
)

# %%
class GuardrailOutput(BaseModel):
    is_relevant: bool = Field(description="Is the question about the document context?")
    reasoning: str = Field(description="Brief reason for the decision")
    
guardrail_system_prompt = """
You are a security filter for a Contract Assistant. 
Analyze the user's question and determine if it is related to contract analysis 
or document retrieval.

Respond ONLY with a JSON object following this structure:
{{
  "is_relevant": true/false,
  "reasoning": "brief explanation"
}}
"""


guardrail_llm = llm.with_structured_output(GuardrailOutput)
guardrail_prompt = ChatPromptTemplate.from_messages([
    ("system", guardrail_system_prompt),
    ("human", "{question}")
])

guardrail_chain = guardrail_prompt | guardrail_llm

# %%

# %%
def process_uploaded_file(file_path):
    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded file is empty or was not written correctly.")

    # Case-insensitive extension check
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embedder,
        collection_name="temp_collection"
    )
    return vectorstore.as_retriever()

# %%
def get_rag_chain(retriever):
    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a contract assistant. Use the context to answer the question."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    return (
        RunnableParallel({
            "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": itemgetter("question"),
            "history": itemgetter("history")
        })
        | contextual_prompt 
        | llm 
        | StrOutputParser()
    )

