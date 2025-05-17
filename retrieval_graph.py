"""
Retrieval Graph for RAG Agent

This module defines a LangGraph for question answering, which includes:
- Processing user queries
- Retrieving relevant documents from Supabase
- Generating responses based on retrieved context
"""

from typing import List, Any, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseRetriever
from langgraph.graph import StateGraph,START,END

# Define the state for the retrieval graph
class RetrievalState(TypedDict):
    question: str
    context: List[Document]
    messages: List[Any]
    response: Optional[str]
    error: Optional[str]

def retrieve_documents(state: RetrievalState, retriever: BaseRetriever):
    """
    Retrieve relevant documents based on the user query.
    
    Args:
        state: The current state
        retriever: The retriever to use
        
    Returns:
        Updated state with retrieved documents
    """
    try:
        question = state["question"]
        # documents = retriever.get_relevant_documents(question)
        documents = retriever.invoke(question)

        # Update state
        state["context"] = documents
        
    except Exception as e:
        state["error"] = str(e)
    
    return state

def generate_response(state: RetrievalState, llm) -> RetrievalState:
    """
    Generate a response based on the retrieved documents.
    
    Args:
        state: The current state
        llm: The language model to use
        
    Returns:
        Updated state with generated response
    """
    try:
        # if not llm:
        #     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Extract context from documents
        context_str = "\n\n".join([doc.page_content for doc in state["context"]])
        
        # Create system message with context
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant that answers questions based on the provided context. "
                "If you don't know the answer based on the context, say that you don't know. "
                "Don't make up information that's not in the context."
            )
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Format prompt
        messages = prompt.format_messages(
            context=context_str,
            messages=[HumanMessage(content=state["question"])]
        )
        
        # Generate response
        response = llm.invoke(messages)
        
        # Update state
        state["response"] = response.content
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content)
        ]
        
    except Exception as e:
        state["error"] = str(e)
    
    return state

def should_continue(state: RetrievalState) -> bool:
    """
    Determine if the conversation should continue.
    
    Args:
        state: The current state
        
    Returns:
        "continue" if there's a new question, "end" otherwise
    """
    if state.get("question") and not state.get("error"):
        return True
    return False

def create_retrieval_graph(retriever: BaseRetriever, llm):
    """
    Create the retrieval graph.
    
    Args:
        retriever: The retriever to use
        llm: The language model to use
        
    Returns:
        A LangGraph for question answering
    """
    # Initialize the graph
    workflow = StateGraph(RetrievalState)
    
    # Add nodes
    workflow.add_node("retrieve", lambda state: retrieve_documents(state, retriever))
    workflow.add_node("generate", lambda state: generate_response(state, llm))

    # Define edges
    workflow.add_edge(START, "retrieve")
    # workflow.add_edge("retrieve", "generate")

    # Add conditional edge
    workflow.add_conditional_edges(
        "retrieve",
        should_continue,
        {
            True: "generate",
            False: END
        }
    )

    workflow.add_edge("generate", END)
    

    
    # Set entry point
    # workflow.set_entry_point("retrieve")
    
    # Compile the graph
    return workflow.compile()

# Define END constant for the graph
# END = "end"

class RAGChatbot:
    """
    A RAG chatbot that uses the retrieval graph to answer questions.
    """
    
    def __init__(self, retriever: BaseRetriever, llm):

        self.retriever = retriever
        self.llm = llm
        self.graph = create_retrieval_graph(retriever, self.llm)
        self.conversation_history = []
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the chatbot.
        
        Args:
            question: The question to ask
            
        Returns:
            The chatbot's response
        """
        # Initialize state
        state = {
            "question": question,
            "context": [],
            "messages": self.conversation_history,
            "response": None,
            "error": None
        }
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Update conversation history
        self.conversation_history = result["messages"]
        
        # Return response
        if result.get("error"):
            return f"Error: {result['error']}"
        return result["response"]