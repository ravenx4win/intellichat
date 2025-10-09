# 🤖 Intellichat - Universal Document Q&A Assistant

**A powerful document Q&A system that works both locally and in the cloud, with AI-powered responses using Hugging Face models and optional web search integration.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Local](https://img.shields.io/badge/100%25-Local-brightgreen.svg)](https://github.com/ravenx4win/intellichat-1)

## 🌟 Key Features

- 🌐 **Universal Deployment** - Works locally and in the cloud
- 🤖 **AI-Powered Q&A** - Uses Hugging Face models for intelligent responses
- 📄 **PDF Document Processing** - Upload and analyze PDF documents
- 🔍 **Web Search Integration** - Optional web search for enhanced responses
- 💾 **SQLite Database** - Persistent storage for documents and chat history
- 🌐 **Cloud Ready** - Deploy on Streamlit Cloud, Heroku, etc.
- 🎨 **Beautiful UI** - Modern Streamlit interface
- ⚡ **Fast & Reliable** - Optimized for performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Hugging Face API key (FREE) - Get it at: https://huggingface.co/settings/tokens

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ravenx4win/intellichat-1.git
   cd intellichat-1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create `.env` file with:
     ```
     HUGGINGFACE_API_KEY=your_huggingface_api_key_here
     ```

5. **Run the application**
   ```bash
   streamlit run app.py --server.port 8502 --server.address 0.0.0.0
   ```

5. **Access your app**
   - Local: `http://localhost:8502`
   - Network: `http://YOUR_IP:8502`

## 📋 Usage

1. **Upload a PDF** - Use the file uploader to add your document
2. **Wait for Processing** - The app extracts text and creates a local knowledge base
3. **Ask Questions** - Type questions in natural language
4. **Get Smart Answers** - Receive intelligent responses based on your document
5. **Continue Conversations** - Ask follow-up questions with full context

## 🔧 Technical Architecture

### Local Components
- **Frontend**: Streamlit web interface
- **Document Processing**: PyPDF2 and pdfplumber for text extraction
- **AI Engine**: Local text matching and intelligent response generation
- **Database**: SQLite for persistent storage
- **Search**: Advanced keyword matching with SQLite indexing

### No External Dependencies
- ❌ No Hugging Face API
- ❌ No OpenAI API
- ❌ No external AI services
- ❌ No internet connection required
- ❌ No API keys or authentication

## 🛡️ Privacy & Security

- **Complete Privacy** - All data stays on your machine
- **No Data Sharing** - Nothing is sent to external services
- **Local Processing** - All AI processing happens locally
- **Secure Storage** - SQLite database stored on your machine
- **Network Control** - Only accessible on your local network

## 🌐 Network Access

### Local Network Sharing
Your app can be accessed by other devices on your network:
- **Find your IP**: `ifconfig | grep "inet " | grep -v 127.0.0.1`
- **Share URL**: `http://YOUR_IP:8502`
- **Access from**: Any PC, phone, or tablet on your WiFi

### Security Features
- Only accessible on your local network
- No public internet exposure
- Complete control over access

## 📁 Project Structure

```
intellichat-1/
├── app_local.py              # Main local application
├── requirements_local.txt     # Minimal dependencies
├── README.md                 # This documentation
├── .gitignore               # Git ignore file
├── intellichat.db           # SQLite database (auto-created)
└── venv/                    # Virtual environment (not included in repo)
```

## 🔧 Configuration

### Customization Options
You can modify these parameters in `app_local.py`:
- **Chunk Size**: `chunk_size=1000` (adjust for document complexity)
- **Chunk Overlap**: `chunk_overlap=200` (for better context)
- **Search Results**: `k=5` (number of relevant chunks)
- **Response Templates**: Customize AI response patterns

### Database Management
- **Auto-created**: SQLite database is created automatically
- **Persistent**: Data survives app restarts
- **Backup**: Copy `intellichat.db` to backup your data
- **Reset**: Delete `intellichat.db` to start fresh

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use a different port
   streamlit run app_local.py --server.port 8503
   ```

2. **PDF Processing Issues**
   - Ensure PDF contains extractable text (not just images)
   - Try with a different PDF file
   - Check if PDF is password-protected

3. **Network Access Issues**
   - Check firewall settings
   - Ensure devices are on same network
   - Verify IP address is correct

4. **Memory Issues**
   - Large documents may require more RAM
   - Consider reducing chunk size for very large documents

### Performance Tips
- Use smaller chunk sizes for faster processing
- SQLite handles large documents efficiently
- Chat history is automatically managed

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit** for the beautiful web interface
- **SQLite** for reliable local database storage
- **PyPDF2 & pdfplumber** for PDF text extraction
- **LangChain** for the text processing framework

## 🆚 Comparison: Local vs Cloud

| Feature | Local Version | Cloud Version |
|---------|---------------|---------------|
| **Privacy** | ✅ 100% Private | ❌ Data sent to external services |
| **API Keys** | ✅ None required | ❌ Requires API keys |
| **Internet** | ✅ Works offline | ❌ Requires internet |
| **Cost** | ✅ Completely free | ❌ May have usage costs |
| **Speed** | ✅ Instant responses | ❌ Network latency |
| **Control** | ✅ Full control | ❌ Dependent on external services |
| **Customization** | ✅ Fully customizable | ❌ Limited by service terms |

## 🎯 Why Choose Local?

- **Privacy First** - Your documents never leave your machine
- **No Dependencies** - No external services to fail
- **Complete Control** - Customize and modify as needed
- **Cost Effective** - No ongoing API costs
- **Reliable** - Works even without internet
- **Secure** - No data breaches or unauthorized access

---

**Perfect for:** Students, researchers, businesses, and anyone who values privacy and wants a reliable, offline document Q&A system.

**Start your local AI journey today!** 🚀📚