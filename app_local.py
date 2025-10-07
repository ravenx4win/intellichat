import os
import tempfile
import gc
import sqlite3
import hashlib
from datetime import datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import PyPDF2
import pdfplumber
import re
import random

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
    
    # Add individual word searches
    for word in query_words:
        if len(word) > 1:  # Include shorter words for better matching
            search_conditions.append('(chunk_text LIKE ? OR keywords LIKE ?)')
            params.extend([f'%{word}%', f'%{word}%'])
    
    # Add synonym/related word searches
    synonyms = {
        'summary': ['overview', 'abstract', 'introduction', 'main points'],
        'about': ['content', 'subject', 'topic', 'discusses'],
        'what': ['content', 'information', 'details'],
        'pdf': ['document', 'file', 'text']
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
    page_title="Intellichat - Local",
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
    .local-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
try:
    init_database()
except Exception as e:
    st.warning(f"Database initialization note: {str(e)}")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None

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

def process_document_with_sqlite(filename: str, text: str):
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

def create_local_ai_response(question, context_chunks):
    """Create AI-like response using local text processing and templates."""
    
    # Extract key information from context
    context_text = "\n".join(context_chunks) if context_chunks else ""
    
    # Simple keyword extraction
    question_words = set(question.lower().split())
    
    # Response templates based on question type
    if any(word in question.lower() for word in ['what', 'what is', 'what are']):
        response_type = "definition"
    elif any(word in question.lower() for word in ['how', 'how to', 'how does']):
        response_type = "explanation"
    elif any(word in question.lower() for word in ['summary', 'summarize', 'overview']):
        response_type = "summary"
    elif any(word in question.lower() for word in ['why', 'why is', 'why does']):
        response_type = "reasoning"
    else:
        response_type = "general"
    
    # Generate response based on type and context
    if context_text:
        # Find relevant sentences
        sentences = context_text.split('. ')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in question_words):
                relevant_sentences.append(sentence.strip())
        
        # Take the most relevant sentences
        relevant_text = '. '.join(relevant_sentences[:3])
        
        if response_type == "definition":
            response = f"""**Answer:** Based on the document content, here's what I found:

{relevant_text}

This information directly relates to your question about "{question}" and is extracted from the document content."""
        
        elif response_type == "explanation":
            response = f"""**Explanation:** Here's how it works according to the document:

{relevant_text}

This provides a detailed explanation related to your question about "{question}"."""
        
        elif response_type == "summary":
            response = f"""**Summary:** Here's an overview of the relevant content:

{relevant_text}

This summarizes the key points related to your question about "{question}"."""
        
        elif response_type == "reasoning":
            response = f"""**Analysis:** Here's the reasoning behind this:

{relevant_text}

This explains the reasoning related to your question about "{question}"."""
        
        else:
            response = f"""**Information Found:** Here's what the document contains:

{relevant_text}

This information is relevant to your question about "{question}"."""
    
    else:
        # No context found - provide helpful response
        response = f"""**No specific information found** about "{question}" in the document.

**Suggestions:**
- Try asking about the main topics in the document
- Ask "What is this document about?"
- Try "What are the key points?"
- Ask "Can you summarize this document?"

The document content may not contain information directly related to your specific question."""
    
    return response

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
st.markdown('<h1 class="main-header">📄 Intellichat – Local Document Q&A Assistant <span class="local-badge">100% LOCAL</span></h1>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Local mode indicator
    st.success("✅ **100% Local Mode** - No external APIs required!")
    st.info("""
    **Local Features:**
    - 📄 PDF document processing
    - 🧠 Intelligent text matching
    - 💾 SQLite database storage
    - 💬 Chat history
    - 🔍 Smart keyword search
    - 🎨 Beautiful UI
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a PDF document
    2. Wait for processing to complete
    3. Ask questions about the document
    4. Get intelligent responses!
    """)
    
    st.markdown("### 🔧 Features")
    st.markdown("""
    - 📄 PDF document processing
    - 🧠 Local AI-powered Q&A
    - 💭 Conversational memory
    - 🔍 Smart text search
    - 🎨 Beautiful UI
    - 🛡️ 100% Private & Local
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
                    document_id = process_document_with_sqlite(pdf_file.name, text)
                    
                    if document_id:
                        st.session_state.current_document_id = document_id
                        st.success(f"✅ Document saved to database with ID: {document_id}")
                        
                        # Clear chat history when new document is uploaded
                        st.session_state.chat_history = []
                        
                        st.success("✅ Local AI assistant ready!")
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
                    # Get relevant context from the document using SQLite
                    context_chunks = search_document_sqlite(st.session_state.current_document_id, question, k=5)
                    
                    # Generate local AI response
                    answer = create_local_ai_response(question, context_chunks)
                    
                    # Save to database and store in session
                    save_chat_history(st.session_state.current_document_id, question, answer)
                    st.session_state.chat_history.append((question, answer))
                    
                    # Display the response immediately
                    st.success("✅ Response generated!")
                    st.markdown("---")
                    st.markdown("### 🤖 Local AI Response:")
                    st.markdown(answer)
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"❌ Error getting response: {str(e)}")
                    st.info("💡 Try asking a simpler question.")
        
        # Display chat history
        display_chat_history()
        
        # Clear chat and reset buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Document"):
                st.session_state.current_document_id = None
                st.session_state.chat_history = []
                st.success("✅ Document reset! Upload a new document to start fresh.")
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
        
        st.success("✅ All data cleared! Upload a new document to start fresh.")
        st.rerun()
    
    else:
        st.info("👆 Please upload a PDF document first to start chatting!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Local AI & SQLite | Built with Streamlit | 100% Private & Local</p>
    </div>
    """,
    unsafe_allow_html=True
)
