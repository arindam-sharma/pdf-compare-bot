import streamlit as st
import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pinecone import Pinecone as PineconeClient
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Custom CSS for modern look    
custom_css = """
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #2d3436;
        background-color: #2d3436;
        color: #ffffff;
    }
    .question {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        color: #000000;
    }
    .answer {
        background-color: #2d3436;
        border-left: 4px solid #6c757d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .summary {
        background-color: #2d3436;
        border-left: 4px solid #28a745;
        color: #ffffff;
    }
    .source-header {
        color: #a8e6cf;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
</style>
"""

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Medical Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
DB1_INDEX_NAME = "usmle-inner-circle"
DB2_INDEX_NAME = "internal-med-surgery"

class RAGChatbot:
    def __init__(self):
        # Initialize environment variables from secrets
        self._initialize_environment()
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
        self.llm = ChatOpenAI(model=LLM_MODEL)
        self.pc = PineconeClient(api_key=os.environ["PINECONE_API_KEY"])
        
        # Initialize vector stores
        self._initialize_vector_stores()
        
        # Initialize prompts with improved system messages
        self._initialize_prompts()

    def _initialize_environment(self):
        """Initialize environment variables from Streamlit secrets"""
        required_secrets = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        for secret in required_secrets:
            if secret not in st.secrets:
                st.error(f"Missing required secret: {secret}")
                st.stop()
            os.environ[secret] = st.secrets[secret]

    def _initialize_vector_stores(self):
        """Initialize Pinecone vector stores with error handling"""
        try:
            self.db1 = Pinecone.from_existing_index(
                index_name=DB1_INDEX_NAME,
                embedding=self.embeddings,
                text_key="text"
            )
            self.db2 = Pinecone.from_existing_index(
                index_name=DB2_INDEX_NAME,
                embedding=self.embeddings,
                text_key="text"
            )
        except Exception as e:
            st.error(f"Failed to initialize vector stores: {str(e)}")
            st.stop()

    def _initialize_prompts(self):
        """Initialize enhanced prompts for better responses"""
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical knowledge assistant specializing in USMLE preparation. 
            Provide clear, structured answers based solely on the given context. Pick the relevant topics to come up and reason your answer. 
            Format your response with:
            - Key points highlighted at the beginning
            - Relevant USMLE-specific details
            - Clinical pearls when applicable
            The primary user is talking to Eishvauk, who is a medical student and you can refer to the user by her Name.
            Use markdown formatting for better readability."""),
            ("user", "Context: {context}\nQuestion: {question}")
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a comprehensive comparison of the information from both sources:
            1. USMLE Inner Circle
            2. Internal Medicine & Surgery
            
            Focus on:
            - Key differences in content coverage
            - Relevance to USMLE Step 2 and 3
            - Practical study recommendations
            - Integration of both sources for optimal preparation
             
            The primary user is talking to Eishvauk, who is a medical student and you can refer to the user by her Name.
            
            Format your response with clear sections and bullet points."""),
            ("user", "Source 1: {db1_answer}\nSource 2: {db2_answer}\nProvide analysis:")
        ])

    def setup_retrieval_chain(self, vectorstore):
        """Set up retrieval chain with improved context handling"""
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 15  # Number of documents to retrieve
            }
        )
        
        chain = (
            RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough()
            })
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    async def get_answers_async(self, question: str) -> Tuple[str, str, str]:
        """Get answers in parallel using asyncio"""
        with st.spinner("🔍 Searching medical knowledge bases..."):
            db1_chain = self.setup_retrieval_chain(self.db1)
            db2_chain = self.setup_retrieval_chain(self.db2)
            
            # Create thread pool for parallel execution
            with ThreadPoolExecutor() as executor:
                # Run both chains in parallel
                loop = asyncio.get_event_loop()
                db1_task = loop.run_in_executor(executor, db1_chain.invoke, question)
                db2_task = loop.run_in_executor(executor, db2_chain.invoke, question)
                
                # Wait for both results
                db1_answer, db2_answer = await asyncio.gather(db1_task, db2_task)
                st.progress(75)
                
                # Get summary
                summary_chain = self.summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({
                    "db1_answer": db1_answer,
                    "db2_answer": db2_answer
                })
                
                return db1_answer, db2_answer, summary

    def get_answers(self, question: str) -> Tuple[str, str, str]:
        """Synchronous wrapper for async get_answers"""
        return asyncio.run(self.get_answers_async(question))

def display_chat_message(message_type: str, content: str, source: str = None):
    """Display styled chat messages"""
    if message_type == "question":
        st.markdown(f'<div class="chat-message question">{content}</div>', unsafe_allow_html=True)
    elif message_type == "answer":
        st.markdown(f'<div class="chat-message answer"><div class="source-header">{source}</div>{content}</div>', unsafe_allow_html=True)
    elif message_type == "summary":
        st.markdown(f'<div class="chat-message summary">{content}</div>', unsafe_allow_html=True)

def main():
    st.title("🏥 Padhai Karo Acche Se")
    st.markdown("""
    *This stuff will give you the results from the two pdf files and will then give you a summary of the results    *
    """)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.info("""
        This assistant combines knowledge from:
        - USMLE Inner Circle
        - Internal Medicine & Surgery
        
        Perfect for USMLE Step 2 & 3 preparation!
        """)
        
        # Add clear chat button
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.rerun()
    
    # Chat input with placeholder
    question = st.text_input(
        "Ask your medical question:",
        placeholder="e.g., What's the diagnostic approach for acute pancreatitis?"
    )
    
    if question:
        try:
            # Get answers and summary
            db1_answer, db2_answer, summary = st.session_state.chatbot.get_answers(question)
            
            # Store in history
            st.session_state.history.append({
                "question": question,
                "db1_answer": db1_answer,
                "db2_answer": db2_answer,
                "summary": summary
            })
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display chat history
    for item in reversed(st.session_state.history):
        display_chat_message("question", f"🤔 {item['question']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_chat_message("answer", item["db1_answer"], "USMLE Inner Circle")
            
        with col2:
            display_chat_message("answer", item["db2_answer"], "Internal Medicine & Surgery")
        
        display_chat_message("summary", f"📋 **Comparative Analysis**\n\n{item['summary']}")
        
        st.markdown("---")

if __name__ == "__main__":
    main()