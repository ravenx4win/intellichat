import os
import tempfile
import gc
import requests
import time
import sqlite3
import hashlib
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import PyPDF2
import pdfplumber
import torch

# Google Gemini AI Model Integration
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_community.llms import HuggingFaceHub
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Load environment variables (local) or Streamlit secrets (cloud)
load_dotenv()

# Try to get API keys from environment variables first (local development)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# If not found in environment, try Streamlit secrets (cloud deployment)
if not GOOGLE_API_KEY:
    try:
        GOOGLE_API_KEY = st.secrets["api_keys"]["GOOGLE_API_KEY"]
    except:
        GOOGLE_API_KEY = None

if not HUGGINGFACE_API_KEY:
    try:
        HUGGINGFACE_API_KEY = st.secrets["api_keys"]["HUGGINGFACE_API_KEY"]
    except:
        HUGGINGFACE_API_KEY = None

# Check for available AI models
available_models = []
if GOOGLE_API_KEY:
    available_models.append("Google Gemini (Free)")
if HUGGINGFACE_API_KEY:
    available_models.append("Hugging Face (Free)")

if not any([GOOGLE_API_KEY, HUGGINGFACE_API_KEY]):
    st.error("⚠️ No AI Models Available!")
    st.info("""
    **Setting up AI Models:**
    
    **Option 1: Google Gemini (Recommended - Free)**
    - Get free API key: https://aistudio.google.com/app/apikey
    - No payment required!
    - Set environment variable: `GOOGLE_API_KEY`
    - Excellent for document analysis and Q&A
    
    **Option 2: Hugging Face (Free API)**
    - Get free API key: https://huggingface.co/settings/tokens
    - Set environment variable: `HUGGINGFACE_API_KEY`
    - Good for basic text generation
    """)
    st.stop()
else:
    st.success(f"✅ Available AI Models: {', '.join(available_models)}")

# SQLite Database Functions
def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect('intellichat.db')
    cursor = conn.cursor()
    
    # Create documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content_hash TEXT UNIQUE,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create chunks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            chunk_text TEXT,
            keywords TEXT,
            chunk_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    ''')
    
    # Create chat_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON chunks(keywords)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_document_id ON chat_history(document_id)')
    
    conn.commit()
    conn.close()

def get_database_connection():
    """Get SQLite database connection."""
    return sqlite3.connect('intellichat.db')

def save_document(filename, content):
    """Save document to database and return document ID."""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Check if document already exists
    cursor.execute('SELECT id FROM documents WHERE content_hash = ?', (content_hash,))
    existing = cursor.fetchone()
    
    if existing:
        conn.close()
        return existing[0]
    
    # Insert new document
    cursor.execute('''
        INSERT INTO documents (filename, content_hash, content)
        VALUES (?, ?, ?)
    ''', (filename, content_hash, content))
    
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return document_id

def save_chunks(document_id, chunks):
    """Save document chunks to database."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Clear existing chunks for this document
    cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
    
    # Insert new chunks
    for i, chunk in enumerate(chunks):
        # Extract keywords from chunk
        keywords = ' '.join(set(chunk.lower().split()))
        cursor.execute('''
            INSERT INTO chunks (document_id, chunk_text, keywords, chunk_index)
            VALUES (?, ?, ?, ?)
        ''', (document_id, chunk, keywords, i))
    
    conn.commit()
    conn.close()

def search_chunks(document_id, query, limit=5):
    """Enhanced search chunks for a document using improved keyword matching."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Enhanced keyword search with better matching
    query_words = query.lower().split()
    search_conditions = []
    params = []
    
    # Add exact phrase search
    if len(query.strip()) > 0:
        search_conditions.append('chunk_text LIKE ?')
        params.append(f'%{query.lower()}%')
    
    # Add individual word searches with scoring
    for word in query_words:
        if len(word) > 1:  # Include shorter words for better matching
            search_conditions.append('(chunk_text LIKE ? OR keywords LIKE ?)')
            params.extend([f'%{word}%', f'%{word}%'])
    
    # Add synonym/related word searches
    synonyms = {
        'summary': ['overview', 'abstract', 'introduction', 'main points'],
        'about': ['content', 'subject', 'topic', 'discusses'],
        'what': ['content', 'information', 'details'],
        'pdf': ['document', 'file', 'text'],
        'how': ['method', 'process', 'procedure', 'steps'],
        'why': ['reason', 'cause', 'purpose', 'because'],
        'when': ['time', 'date', 'schedule', 'timeline']
    }
    
    for word in query_words:
        if word in synonyms:
            for synonym in synonyms[word]:
                search_conditions.append('(chunk_text LIKE ? OR keywords LIKE ?)')
                params.extend([f'%{synonym}%', f'%{synonym}%'])
    
    if not search_conditions:
        # If no meaningful words, return all chunks
        cursor.execute('''
            SELECT chunk_text FROM chunks 
            WHERE document_id = ? 
            ORDER BY chunk_index 
            LIMIT ?
        ''', (document_id, limit))
    else:
        # Build the query with OR conditions for better matching
        where_clause = ' OR '.join(search_conditions)
        params = [document_id] + params + [limit]
        
        cursor.execute(f'''
            SELECT chunk_text FROM chunks 
            WHERE document_id = ? AND ({where_clause})
            ORDER BY chunk_index 
            LIMIT ?
        ''', params)
    
    results = cursor.fetchall()
    conn.close()
    
    return [result[0] for result in results]

def save_chat_history(document_id, question, answer):
    """Save chat history to database."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO chat_history (document_id, question, answer)
        VALUES (?, ?, ?)
    ''', (document_id, question, answer))
    
    conn.commit()
    conn.close()

def get_chat_history(document_id, limit=10):
    """Get chat history for a document."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT question, answer, created_at FROM chat_history 
        WHERE document_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (document_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    
    return [(result[0], result[1]) for result in results]

def get_document_info(document_id):
    """Get document information."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT filename, created_at FROM documents WHERE id = ?
    ''', (document_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    return result

def clear_document_data(document_id):
    """Clear all data for a specific document."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM chat_history WHERE document_id = ?', (document_id,))
    cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
    cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
    
    conn.commit()
    conn.close()

def get_all_documents():
    """Get all documents from database."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, created_at FROM documents 
        ORDER BY created_at DESC
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    return results

# Set up Streamlit UI
st.set_page_config(
    page_title="Intellichat",
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

# Initialize database (with error handling for Streamlit Cloud)
try:
    init_database()
except Exception as e:
    st.warning(f"Database initialization note: {str(e)}")
    # Continue without database if there are issues

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "use_fallback" not in st.session_state:
    st.session_state.use_fallback = False

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

def process_document_with_sqlite(filename: str, text: str, api_key: str):
    """Process document and save to SQLite database."""
    if not text:
        return None

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
        return None
    
    try:
        # Save document to database
        document_id = save_document(filename, text)
        
        # Save chunks to database
        save_chunks(document_id, chunks)
        
        st.success(f"✅ Document '{filename}' processed and saved to database!")
        return document_id
        
    except Exception as e:
        st.error(f"❌ Error saving document to database: {str(e)}")
        return None

def create_qa_chain(vectorstore, api_key: str):
    """Create the conversational retrieval chain using Hugging Face."""
    if not vectorstore:
        return None
    
    try:
        # Use a state-of-the-art model for superior responses
        with st.spinner("🔄 Loading state-of-the-art AI model (this may take a moment)..."):
            # Load a modern, high-quality model for sophisticated responses
            model_name = "microsoft/DialoGPT-large"  # Keep DialoGPT for now, but we'll enhance it
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Set padding token
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                low_cpu_mem_usage=True  # Use less CPU memory during loading
            )
            
            # Create optimized pipeline with proper token handling
            pipe = pipeline(
                "text-generation",  # Changed to text-generation for causal models
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,  # Use max_new_tokens instead of max_length
                min_length=20,       # Minimum response length
                temperature=0.7,     # Balanced temperature for natural responses
                do_sample=True,      # Enable sampling for more natural responses
                top_p=0.9,           # Nucleus sampling for quality
                repetition_penalty=1.1,  # Prevent repetition
                pad_token_id=tokenizer.eos_token_id,  # Proper padding
                truncation=True,     # Enable truncation for long inputs
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            # Create HuggingFace pipeline wrapper
            llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create memory with explicit output key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Set output key for memory
        )
        
        # Create the chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Reduced chunks for efficiency
            memory=memory,
            return_source_documents=False,  # Disable source documents to avoid multiple keys
            output_key="answer"  # Explicitly set the output key for memory
        )
        
        st.success("✅ AI model loaded successfully!")
        cleanup_memory()  # Clean up memory after loading
        return qa_chain
        
    except Exception as e:
        st.warning(f"AI model failed to load: {str(e)}")
        st.info("🔄 Using intelligent text matching instead...")
        return None  # Fallback will be handled in main logic

def cleanup_memory():
    """Clean up memory to prevent crashes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preprocess_question(question):
    """Preprocess question to prevent model issues."""
    # Clean and format the question
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    return question

def truncate_text_intelligently(text, max_tokens=400):
    """Truncate text intelligently to fit within token limits."""
    if not text:
        return text
    
    # Split into sentences for better truncation
    sentences = text.split('. ')
    truncated = ""
    token_count = 0
    
    for sentence in sentences:
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = len(sentence) // 4
        if token_count + estimated_tokens > max_tokens:
            break
        truncated += sentence + ". "
        token_count += estimated_tokens
    
    return truncated.strip()

def search_web(query, max_results=3):
    """Search the web for additional information."""
    try:
        # Simple web search using DuckDuckGo (no API key required)
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(search_url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            # Extract relevant information
            if 'Abstract' in data and data['Abstract']:
                results.append(f"Web Information: {data['Abstract']}")
            
            if 'RelatedTopics' in data:
                for topic in data['RelatedTopics'][:max_results]:
                    if 'Text' in topic:
                        results.append(f"Related: {topic['Text']}")
            
            return "\n".join(results) if results else ""
        else:
            return ""
    except Exception as e:
        st.warning(f"Web search failed: {str(e)}")
        return ""

def create_sophisticated_prompt(question, context="", web_info=""):
    """Create a sophisticated prompt for better AI responses with web integration."""
    # Truncate context to prevent token overflow
    truncated_context = truncate_text_intelligently(context, max_tokens=200)
    truncated_web = truncate_text_intelligently(web_info, max_tokens=200)
    
    prompt = f"""You are an intelligent AI assistant with access to document content and web information. Please provide a comprehensive, accurate, and helpful response to the user's question.

Document Context: {truncated_context if truncated_context else "No specific document context available"}

Web Information: {truncated_web if truncated_web else "No additional web information available"}

User Question: {question}

Please provide a detailed, accurate, and well-structured response based on the available information. Combine document content with web information when relevant. If the information is not available, please say so clearly and suggest what the user might ask instead.

Response:"""
    return prompt

def is_repetitive_response(response):
    """Check if response is repetitive (like 'Helpful Helpful Helpful...')."""
    if not response or len(response) < 10:
        return False
    
    # Split into words and check for repetition
    words = response.split()
    if len(words) < 2:
        return False
    
    # Check if the same word appears too many times
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # If any word appears more than 20% of the time, it's repetitive
    max_count = max(word_counts.values())
    if max_count > len(words) * 0.2:
        return True
    
    # Check for specific problematic patterns
    response_lower = response.lower()
    if any(pattern in response_lower for pattern in ['helpful', 'assistant', 'ai', 'model']):
        if response_lower.count('helpful') > 1 or response_lower.count('assistant') > 1:
            return True
    
    return False

def search_document_sqlite(document_id, question, k=5):
    """Search document using SQLite database."""
    if not document_id:
        return []
    
    try:
        # Use SQLite search function
        chunks = search_chunks(document_id, question, k)
        return chunks
    except Exception as e:
        st.warning(f"Search error: {str(e)}")
        return []

def create_gemini_ai_qa(document_id):
    """Create AI-powered Q&A using Google Gemini (superior quality)."""
    def gemini_ai_qa(question):
        try:
            # Get relevant documents using enhanced SQLite search
            relevant_chunks = search_document_sqlite(document_id, question, k=8)
            
            if relevant_chunks:
                # Combine relevant content
                context = "\n\n".join(relevant_chunks)
                
                # Create sophisticated prompt for AI
                prompt = f"""You are an expert AI assistant analyzing a document. Please provide a comprehensive, accurate, and helpful response to the user's question.

DOCUMENT CONTEXT:
{context[:2000]}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the document context carefully
2. Provide a detailed, accurate response based on the document content
3. If the information is not available in the document, clearly state this
4. Structure your response clearly with proper formatting
5. Be specific and cite relevant parts of the document
6. If the question is about a specific topic, focus on that topic
7. Provide actionable insights when possible

RESPONSE:"""
                
                # Use Google Gemini (Primary Choice - Superior Quality)
                if GOOGLE_API_KEY:
                    try:
                        if GEMINI_AVAILABLE:
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-pro",
                                temperature=0.7,
                                max_output_tokens=1000,
                                google_api_key=GOOGLE_API_KEY
                            )
                            response = llm.invoke(prompt).content
                            return {"answer": response}
                        else:
                            st.warning("Google Gemini library not installed. Install with: pip install langchain-google-genai")
                    except Exception as e:
                        st.warning(f"Google Gemini not available: {e}")
                
                # Fallback to Hugging Face (Free)
                if HUGGINGFACE_API_KEY:
                    try:
                        llm = HuggingFaceHub(
                            repo_id="microsoft/DialoGPT-medium",
                            huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                            model_kwargs={"temperature": 0.7, "max_length": 500}
                        )
                        response = llm.invoke(prompt)
                        return {"answer": response}
                    except Exception as e:
                        st.warning(f"Hugging Face API not available: {e}")
                
                # Fallback to intelligent text matching
                response = f"""**AI Analysis:**

Based on the document content, here's what I found:

{context[:1000]}

**Answer:** This information is extracted from your document and should help answer your question about "{question}". The content appears to be relevant to your query."""
                
                return {"answer": response}
            else:
                # Try a broader search for general questions
                if any(word in question.lower() for word in ['summary', 'about', 'what', 'main', 'overview', 'content']):
                    # Get all chunks for general questions
                    conn = get_database_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT chunk_text FROM chunks 
                        WHERE document_id = ? 
                        ORDER BY chunk_index 
                        LIMIT 5
                    ''', (document_id,))
                    all_chunks = cursor.fetchall()
                    conn.close()
                    
                    if all_chunks:
                        content = "\n\n".join([chunk[0] for chunk in all_chunks])
                        
                        # Use AI for better overview responses
                        if GOOGLE_API_KEY:
                            if GEMINI_AVAILABLE:
                                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
                                prompt = f"Provide a comprehensive overview of this document content: {content[:1500]}. User asked: {question}"
                                response = llm.invoke(prompt).content
                            else:
                                st.warning("Google Gemini library not installed. Install with: pip install langchain-google-genai")
                        elif HUGGINGFACE_API_KEY:
                            if HUGGINGFACE_AVAILABLE:
                                llm = HuggingFaceHub(
                                    repo_id="microsoft/DialoGPT-medium",
                                    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                                    model_kwargs={"temperature": 0.7, "max_length": 500}
                                )
                                prompt = f"Provide a comprehensive overview of this document content: {content[:1500]}. User asked: {question}"
                                response = llm.invoke(prompt)
                            else:
                                st.warning("Hugging Face library not installed. Install with: pip install langchain-community")
                        else:
                            response = f"""**Document Overview:** Here's what the document contains:

{content[:1200]}

This provides an overview of the document content that should help answer your question about "{question}"."""
                    else:
                        response = f"""**Document Content Not Found**

I couldn't find any content in the document. This might be because:
- The PDF contains only images (no extractable text)
- The document is password-protected
- The text extraction failed

**Try uploading a different PDF with extractable text.**"""
                else:
                    response = f"""**No specific information found**

I couldn't find specific information about '{question}' in the document. 

**Try asking about:**
- What is this document about?
- What are the main topics?
- What are the key points?
- Can you summarize this document?
- What does this document discuss?"""
            
            return {"answer": response}
            
        except Exception as e:
            return {"answer": f"Error processing question: {str(e)}"}
    
    return advanced_qa

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
                    <strong>Intellichat:</strong> {answer}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Main UI
st.markdown('<h1 class="main-header">📄 Intellichat – Document Q&A Assistant</h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Model Status
    st.markdown("### 🤖 AI Model Configuration")
    
    if GOOGLE_API_KEY:
        st.success("✅ **Google Gemini Pro** - Premium AI Model!")
        st.info("🚀 Using Google Gemini for superior document analysis and Q&A!")
        primary_model = "Google Gemini Pro"
    elif HUGGINGFACE_API_KEY:
        st.success("✅ **Hugging Face (Free)** - Open Source AI Model")
        st.info("💡 Using Hugging Face as fallback - Good for basic text generation")
        primary_model = "Hugging Face (Free)"
    else:
        st.error("❌ No AI models configured")
        st.info("Please set up at least one AI service")
        st.stop()
    
    st.markdown(f"**Active Model:** {primary_model}")
    st.markdown("💰 **Cost:** $0.00 - Free Tier Available!")
    st.markdown("🎯 **Quality:** Superior AI responses!")
    
    # Web search toggle
    enable_web_search = st.checkbox(
        "🌐 Enable Web Search",
        value=True,
        help="Search the web for additional information to enhance responses"
    )
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Wait for processing to complete
    3. Ask questions about the document
    4. Enjoy intelligent conversations!
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - 📄 PDF document processing
    - 🧠 AI-powered Q&A
    - 💭 Conversational memory
    - 🔍 Semantic search
    - 🎨 Beautiful UI
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
                
                # Process document with SQLite
                with st.spinner("🧠 Saving to database..."):
                    document_id = process_document_with_sqlite(pdf_file.name, text, api_key)
                    
                    if document_id:
                        st.session_state.current_document_id = document_id
                        st.success(f"✅ Document saved to database with ID: {document_id}")
                        
                        # Clear chat history when new document is uploaded
                        st.session_state.chat_history = []
                        st.session_state.use_fallback = False  # Reset fallback preference
                        
                        # Create SQLite QA system
                        with st.spinner("🔗 Setting up AI assistant..."):
                            qa_chain = create_sqlite_qa_fallback(document_id)
                            if qa_chain:
                                st.session_state.qa_chain = qa_chain
                                st.success("✅ AI assistant ready!")
                            else:
                                st.error("❌ Failed to create AI assistant")
                    else:
                        st.error("❌ Failed to save document to database")
            else:
                st.error("❌ Could not extract text from PDF")

with col2:
    st.header("💬 Chat with Your Document")
    
    if hasattr(st.session_state, 'current_document_id') and st.session_state.current_document_id is not None:
        # Chat input
        question = st.text_input(
            "Ask a question about your document:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        if st.button("🚀 Ask Question", type="primary") and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Preprocess the question and get context
                    processed_question = preprocess_question(question)
                    
                    # Get relevant context from the document using SQLite
                    context = ""
                    if hasattr(st.session_state, 'current_document_id') and st.session_state.current_document_id:
                        try:
                            relevant_chunks = search_document_sqlite(st.session_state.current_document_id, question, k=3)
                            context = "\n".join(relevant_chunks)
                        except:
                            context = ""
                    
                    # Get web information for enhanced responses
                    web_info = ""
                    if enable_web_search:
                        try:
                            with st.spinner("🌐 Searching web for additional information..."):
                                web_info = search_web(processed_question)
                        except:
                            web_info = ""
                    
                    # Use SQLite-based document search for reliable responses
                    st.info("🔄 Using intelligent text matching with SQLite database...")
                    
                    # Debug: Show document info and search results
                    if st.session_state.current_document_id:
                        doc_info = get_document_info(st.session_state.current_document_id)
                        if doc_info:
                            st.info(f"📄 Document: {doc_info[0]} (ID: {st.session_state.current_document_id})")
                        
                        # Show search debug info
                        try:
                            debug_chunks = search_document_sqlite(st.session_state.current_document_id, question, k=3)
                            if debug_chunks:
                                st.info(f"🔍 Found {len(debug_chunks)} relevant chunks")
                                st.info(f"📝 First chunk preview: {debug_chunks[0][:100]}...")
                            else:
                                st.warning("⚠️ No relevant chunks found - trying broader search...")
                        except Exception as e:
                            st.warning(f"Search debug error: {str(e)}")
                    
                    # Use Gemini AI for superior responses
                    gemini_ai_qa = create_gemini_ai_qa(st.session_state.current_document_id)
                    response = gemini_ai_qa(processed_question)
                    answer = response["answer"]
                    
                    # Save to database and store in session
                    save_chat_history(st.session_state.current_document_id, question, answer)
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display the response immediately
                    st.success("✅ Response generated!")
                    st.markdown("---")
                    st.markdown("### 🤖 AI Response:")
                    st.markdown(answer)
                    st.markdown("---")
                    
                    # Don't rerun immediately - let user see the response
                    # st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
                    st.info("💡 Try asking a simpler question or check your API key.")
        
        # Display chat history
        display_chat_history()
        
        # Clear chat and reset buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset AI Model"):
                st.session_state.use_fallback = False
                st.session_state.qa_chain = None
                st.success("✅ AI model reset! Try asking a question to reload it.")
                st.rerun()
    
    # Additional cleanup button
    if st.button("🗑️ Clear All Data & Start Fresh", type="secondary"):
        # Clear current document from database if exists
        if hasattr(st.session_state, 'current_document_id') and st.session_state.current_document_id:
            try:
                clear_document_data(st.session_state.current_document_id)
                st.info(f"🗑️ Cleared document data for ID: {st.session_state.current_document_id}")
            except Exception as e:
                st.warning(f"Warning: Could not clear database data: {str(e)}")
        
        # Clear all session state
        st.session_state.chat_history = []
        st.session_state.current_document_id = None
        st.session_state.qa_chain = None
        st.session_state.use_fallback = False
        
        st.success("✅ All data cleared! Upload a new document to start fresh.")
        st.rerun()
    
    else:
        st.info("👆 Please upload a PDF document first to start chatting!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Hugging Face & LangChain | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
