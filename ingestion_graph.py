"""
Ingestion Graph for RAG Agent

This module defines a LangGraph for document ingestion, which includes:
- Loading documents from various sources
- Splitting documents into chunks
- Generating embeddings for document chunks
- Storing embeddings in Supabase vector database
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase.client import Client, create_client
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
load_dotenv()

# Define the state for the ingestion graph
class IngestionState(TypedDict):
    # def __init__(self):
    #     self.documents: List[Document] = []
    #     self.chunks: List[Document] = []
    #     self.status: str = "idle"
    #     self.error: Optional[str] = None
    #     self.metadata: Dict[str, Any] = {}
    file_path: str
    documents: List[Document]
    chunks: List[Document]
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]

# def load_documents(state: IngestionState, file_path: str) -> IngestionState:
def load_documents(state: IngestionState):
    try:
        file_path = state["file_path"]
        file_path = os.path.normpath(file_path)
        print(file_path)
        if os.path.isfile(file_path):
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            documents = loader.load()
        elif os.path.isdir(file_path):
            loader = DirectoryLoader(file_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file path: {file_path}")
        
        state["documents"] = documents
        state["status"] = "documents_loaded"
        state["metadata"]["document_count"] = len(state["documents"])
        
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "error"
    
    return state

def split_documents(state: IngestionState):

    try:
        if not state["documents"]:
            raise ValueError("No documents to split")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        state["chunks"] = text_splitter.split_documents(state["documents"])
        state["status"] = "documents_split"
        state["metadata"]["chunk_count"] = len(state["chunks"])
        print("Splitting documents successfully with state values",state["status"],state["metadata"]["chunk_count"])

    except Exception as e:
        state["error"] = str(e)
        state["status"] = "error"
    
    return state

def embed_and_store(
    state: IngestionState,
    #     config: RunnableConfig,
    # #     supabase_url: str,
    # # supabase_key: str,
    # # table_name: str = "documents1",
    # # query_name: str = "match_documents5"
):

    try:
        if not state["chunks"]:
            raise ValueError("No document chunks to embed")

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        # Initialize Supabase client
        supabase_client: Client = create_client(supabase_url, supabase_key)
        
        # Initialize embeddings model
        embeddings = OpenAIEmbeddings()
        
        # Store documents in Supabase
        vector_store = SupabaseVectorStore.from_documents(
            documents=state["chunks"],
            embedding=embeddings,
            client=supabase_client,
            table_name="documents1",
            query_name="match_documents5"
        )
        
        state["status"] = "embeddings_stored"
        state["metadata"]["vector_store"] = {
            "type": "supabase",
            "table_name": "documents1",
            "query_name": "match_documents5"
        }
        print("Embeddings stored successfully with state values",state["status"],state["metadata"]["vector_store"])
        
    except Exception as e:
        state["error"] = str(e)
        state["status"] = "error"
    
    return state

def create_ingestion_graph():

    # Initialize the graph
    workflow = StateGraph(IngestionState)
    
    # Add nodes
    workflow.add_node("load_documents", load_documents)
    workflow.add_node("split_documents", split_documents)
    workflow.add_node("embed_and_store", embed_and_store)
    
    # Define edges
    workflow.add_edge(START,"load_documents")
    workflow.add_edge("load_documents", "split_documents")
    workflow.add_edge("split_documents", "embed_and_store")
    workflow.add_edge("embed_and_store", END)
    
    # Compile the graph
    return workflow.compile()

def get_retriever(
    supabase_url: str,
    supabase_key: str,
    table_name: str = "documents1",
    query_name: str = "match_documents5"
):

    # Initialize Supabase client
    supabase_client: Client = create_client(supabase_url, supabase_key)
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=table_name,
        query_name=query_name
    )
    
    # Return retriever
    return vector_store.as_retriever()