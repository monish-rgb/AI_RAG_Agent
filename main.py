"""
Main application for RAG Agent

This module provides a simple interface for using the RAG agent,
combining both the ingestion and retrieval components using Langgraph, Langchain and supabase as vector database.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from ingestion_graph import create_ingestion_graph,get_retriever
from retrieval_graph import RAGChatbot
from IPython.display import Image, display

# Load environment variables
load_dotenv()

class RAGAgent:

    def __init__(
        self,
        table_name: str = "documents1",
        query_name: str = "match_documents5",
        model_name: str = "gpt-4o-mini"
    ):

        # Set API keys from environment variables if not provided
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and API key must be provided")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key must be provided")
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Store configuration
        self.table_name = table_name
        self.query_name = query_name

        # Initialize ingestion graph
        self.ingestion_graph = create_ingestion_graph()
        
        # Initialize retriever and chatbot
        self.retriever = None
        self.chatbot = None
        
        # Try to initialize retriever and chatbot if documents are already ingested
        try:
            self.retriever = get_retriever(
                self.supabase_url,
                self.supabase_key,
                self.table_name,
                self.query_name
            )
            self.chatbot = RAGChatbot(self.retriever, self.llm)
        except Exception:
            # No documents ingested yet, will initialize later
            pass
    
    def ingest_documents(self, file_path: str):
        # Initialize state as a dictionary
        init_state = {
            "file_path": file_path,
            "documents": [],
            "chunks": [],
            "status": "not_started",
            "error": None,
            "metadata": {}
        }

        state = self.ingestion_graph.invoke(init_state)

        # Initialize retriever and chatbot if not already initialized
        if not self.retriever or not self.chatbot:
            self.retriever = get_retriever(
                self.supabase_url,
                self.supabase_key,
                self.table_name,
                self.query_name
            )
            print("Retriever initialized",self.retriever)
            print("Chatbot initialized",self.chatbot)
            self.chatbot = RAGChatbot(self.retriever, self.llm)
        
        # Return status
        return {
            "status": state["status"],
            "error": state["error"],
            "metadata": state["metadata"]
        }
    
    def ask(self, question: str) -> str:
        if not self.chatbot:
            return "Error: No documents have been ingested yet. Please ingest documents first."

        return self.chatbot.ask(question)

    def get_conversation_history(self) -> List:
        if not self.chatbot:
            return []

        return self.chatbot.conversation_history

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in a .env file or in your environment.")
        return
    
    # Initialize RAG agent
    agent = RAGAgent()

    # Example usage
    print("RAG Agent Demo")
    print("==============")
    print("1. Ingest documents")
    print("2. Ask a question")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            file_path = input("Enter the path to the file or directory to ingest: ")
            print(f"Ingesting documents from {file_path}...")
            result = agent.ingest_documents(file_path)
            print(f"Ingestion status: {result['status']}")
            if result.get("error"):
                print(f"Error: {result['error']}")
            else:
                print(f"Ingested {result['metadata'].get('document_count', 0)} documents")
                print(f"Created {result['metadata'].get('chunk_count', 0)} chunks")
        
        elif choice == "2":
            question = input("Enter your question: ")
            print("Thinking...")
            response = agent.ask(question)
            print(f"Response: {response}")
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
