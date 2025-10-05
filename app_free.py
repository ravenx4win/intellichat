import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import PyPDF2
import pdfplumber
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(
    page_title="Intellichat FREE", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .chat-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    .bot-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using multiple methods for better accuracy."""
    text = ""
    
    # Method 1: Try pdfplumber first (better for complex layouts)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        with pdfplumber.open(tmp_file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        os.unlink(tmp_file_path)
        
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}. Trying PyPDF2...")
        
        # Method 2: Fallback to PyPDF2
        try:
            pdf_file.seek(0)  # Reset file pointer
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e2:
            st.error(f"Both PDF extraction methods failed: {e2}")
            return ""
    
    return text.strip()

def create_vectorstore(text: str):
    """Create vector store from text chunks using FREE Hugging Face embeddings."""
    if not text:
        return None, None
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.error("No text chunks could be created from the PDF.")
        return None, None
    
    # Use FREE Hugging Face embeddings
    with st.spinner("🔄 Loading FREE embeddings model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU to avoid GPU issues
        )
    
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_free"
    )
    
    return vectorstore, chunks

def create_qa_chain(vectorstore):
    """Create the conversational retrieval chain using FREE model."""
    if not vectorstore:
        return None
    
    # Use a simple FREE model for Q&A
    with st.spinner("🔄 Loading FREE language model..."):
        try:
            # Use a lightweight model
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
            
            # Create HuggingFace pipeline wrapper
            llm = HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            st.warning(f"Could not load model: {e}. Using simple text matching instead.")
            # Fallback to simple text matching
            return create_simple_qa_chain(vectorstore)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain

def create_simple_qa_chain(vectorstore):
    """Fallback simple Q&A without LLM."""
    def simple_qa(question):
        # Get relevant documents
        docs = vectorstore.similarity_search(question, k=3)
        
        # Simple response based on retrieved content
        if docs:
            content = "\n".join([doc.page_content for doc in docs])
            response = f"Based on the document, here's what I found:\n\n{content[:500]}..."
        else:
            response = "I couldn't find relevant information in the document."
        
        return {"answer": response}
    
    return simple_qa

def display_chat_history():
    """Display the chat history in a nice format."""
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="chat-container">
                <div class="user-message">
                    <strong>You:</strong> {question}
                </div>
                <div class="bot-message">
                    <strong>Intellichat FREE:</strong> {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Main UI
st.markdown('<h1 class="main-header">🆓 Intellichat FREE – Document Q&A Assistant</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### 🆓 FREE Version")
    st.markdown("""
    This version uses:
    - **Hugging Face** embeddings (FREE)
    - **Local models** (FREE)
    - **No API keys** required
    - **No billing** issues
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Wait for processing to complete
    3. Ask questions about the document
    4. Enjoy FREE conversations!
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - 📄 PDF document processing
    - 🧠 AI-powered Q&A (FREE)
    - 💭 Conversational memory
    - 🔍 Semantic search
    - 🎨 Beautiful UI
    - 🆓 No API costs
    """)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Upload Document")
    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to start asking questions"
    )
    
    if pdf_file is not None:
        st.success(f"✅ File uploaded: {pdf_file.name}")
        
        # Process PDF
        with st.spinner("🔄 Processing PDF..."):
            text = extract_text_from_pdf(pdf_file)
            
            if text:
                st.success(f"✅ Extracted {len(text)} characters from PDF")
                
                # Create vector store
                with st.spinner("🧠 Creating knowledge base..."):
                    vectorstore, chunks = create_vectorstore(text)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"✅ Created knowledge base with {len(chunks)} chunks")
                        
                        # Create QA chain
                        with st.spinner("🔗 Setting up FREE AI assistant..."):
                            qa_chain = create_qa_chain(vectorstore)
                            if qa_chain:
                                st.session_state.qa_chain = qa_chain
                                st.success("✅ FREE AI assistant ready!")
                            else:
                                st.error("❌ Failed to create AI assistant")
                    else:
                        st.error("❌ Failed to create knowledge base")
            else:
                st.error("❌ Could not extract text from PDF")

with col2:
    st.header("💬 Chat with Your Document")
    
    if st.session_state.qa_chain is not None:
        # Chat input
        question = st.text_input(
            "Ask a question about your document:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        if st.button("🚀 Ask Question", type="primary") and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response from the chain
                    if callable(st.session_state.qa_chain):
                        # Simple QA function
                        response = st.session_state.qa_chain(question)
                        answer = response["answer"]
                    else:
                        # Full QA chain
                        response = st.session_state.qa_chain({"question": question})
                        answer = response["answer"]
                    
                    # Store in chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Clear the input
                    st.session_state.question_input = ""
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
        
        # Display chat history
        display_chat_history()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("👆 Please upload a PDF document first to start chatting!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🆓 Powered by Hugging Face & LangChain | Built with Streamlit | 100% FREE</p>
    </div>
    """,
    unsafe_allow_html=True
)
