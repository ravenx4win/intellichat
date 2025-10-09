# 🤖 Intellichat - Universal Document Q&A Assistant

**A powerful document Q&A system that works both locally and in the cloud, with AI-powered responses using Google Gemini Pro and Hugging Face models.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-brightgreen.svg)](https://intellichat-fc9eiw4kungqc7ew39af7r.streamlit.app/)

## 🌟 Key Features

- 🌐 **Universal Deployment** - Works locally and in the cloud
- 🚀 **Google Gemini Pro** - Primary AI model for superior responses
- 💡 **Hugging Face Fallback** - Reliable backup AI model
- 📄 **PDF Document Processing** - Upload and analyze PDF documents
- 🔍 **Web Search Integration** - Optional web search for enhanced responses
- 💾 **SQLite Database** - Persistent storage for documents and chat history
- 🌐 **Cloud Ready** - Deploy on Streamlit Cloud, Heroku, etc.
- 🎨 **Beautiful UI** - Modern Streamlit interface
- ⚡ **Fast & Reliable** - Optimized for performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (FREE) - Get it at: https://aistudio.google.com/app/apikey
- Hugging Face API key (FREE) - Get it at: https://huggingface.co/settings/tokens

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ravenx4win/intellichat.git
   cd intellichat
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
     GOOGLE_API_KEY=your_gemini_api_key_here
     HUGGINGFACE_API_KEY=your_huggingface_api_key_here
     ```

5. **Run the application**
   ```bash
   streamlit run app.py --server.port 8502 --server.address 0.0.0.0
   ```

6. **Access your app**
   - Local: `http://localhost:8502`
   - Network: `http://YOUR_IP:8502`
   - Cloud: https://intellichat-fc9eiw4kungqc7ew39af7r.streamlit.app/

## 📋 Usage

1. **Upload a PDF** - Use the file uploader to add your document
2. **Wait for Processing** - The app extracts text and creates a local knowledge base
3. **Ask Questions** - Type questions in natural language
4. **Get Smart Answers** - Receive intelligent responses from Google Gemini Pro
5. **Continue Conversations** - Ask follow-up questions with full context

## 🤖 AI Models

### Primary: Google Gemini Pro
- **Quality**: 🚀 Superior AI responses
- **Cost**: 💰 Completely FREE
- **Features**: Advanced document analysis, contextual understanding
- **Best For**: Complex questions, detailed analysis

### Fallback: Hugging Face
- **Quality**: 💡 Good AI responses  
- **Cost**: 💰 Completely FREE
- **Features**: Reliable text generation
- **Best For**: Basic questions, backup responses

### Emergency: Text Matching
- **Quality**: 📝 Basic responses
- **Cost**: 💰 Completely FREE
- **Features**: Always works, no API required
- **Best For**: Simple queries, offline mode

## 🔧 Technical Architecture

### AI Engine
- **Primary**: Google Gemini Pro (via LangChain)
- **Fallback**: Hugging Face DialoGPT (via LangChain)
- **Emergency**: Intelligent text matching
- **Database**: SQLite for document storage and chat history

### Document Processing
- **PDF Extraction**: PyPDF2 and pdfplumber
- **Text Chunking**: RecursiveCharacterTextSplitter
- **Search**: Advanced SQLite-based document search
- **Storage**: Persistent SQLite database

### Deployment Options
- **Local**: Run on your machine with `.env` file
- **Cloud**: Deploy on Streamlit Cloud with secrets
- **Network**: Share with other devices on your network

## 🌐 Cloud Deployment

### Streamlit Cloud (Recommended)
1. **Fork this repository**
2. **Go to**: https://share.streamlit.io
3. **Connect your GitHub account**
4. **Deploy from**: `ravenx4win/intellichat`
5. **Add secrets** in Streamlit Cloud dashboard:
   ```
   GOOGLE_API_KEY = your_gemini_api_key
   HUGGINGFACE_API_KEY = your_huggingface_api_key
   ```

### Other Cloud Platforms
- **Heroku**: Use Procfile and requirements.txt
- **Railway**: Direct GitHub deployment
- **Render**: Web service deployment
- **AWS/GCP/Azure**: Container deployment

## 📁 Project Structure

```
intellichat/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # Cloud deployment secrets
├── .env                     # Local environment variables
├── setup_gemini.py          # API key setup script
├── README.md               # This documentation
├── .gitignore              # Git ignore file
├── intellichat.db          # SQLite database (auto-created)
└── venv/                   # Virtual environment (not included in repo)
```

## 🔧 Configuration

### API Keys Setup

#### Option 1: Environment Variables (Local)
Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

#### Option 2: Setup Script
Run the setup script:
```bash
python setup_gemini.py
```

#### Option 3: Streamlit Secrets (Cloud)
Add to Streamlit Cloud secrets:
```
[api_keys]
GOOGLE_API_KEY = "your_gemini_api_key"
HUGGINGFACE_API_KEY = "your_huggingface_api_key"
```

### Customization Options
You can modify these parameters in `app.py`:
- **Chunk Size**: `chunk_size=1000` (adjust for document complexity)
- **Chunk Overlap**: `chunk_overlap=200` (for better context)
- **Search Results**: `k=8` (number of relevant chunks)
- **AI Temperature**: `temperature=0.7` (response creativity)

## 🛡️ Privacy & Security

### Local Deployment
- **Complete Privacy** - All data stays on your machine
- **No Data Sharing** - Nothing is sent to external services
- **Local Processing** - All AI processing happens locally
- **Secure Storage** - SQLite database stored on your machine

### Cloud Deployment
- **API Keys Only** - Only API keys are stored in secrets
- **Document Processing** - Happens on cloud servers
- **AI Responses** - Generated by Google/Hugging Face APIs
- **Data Control** - You control what documents to upload

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

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Check if API keys are set
   echo $GOOGLE_API_KEY
   echo $HUGGINGFACE_API_KEY
   ```

2. **Port Already in Use**
   ```bash
   # Use a different port
   streamlit run app.py --server.port 8503
   ```

3. **PDF Processing Issues**
   - Ensure PDF contains extractable text (not just images)
   - Try with a different PDF file
   - Check if PDF is password-protected

4. **Network Access Issues**
   - Check firewall settings
   - Ensure devices are on same network
   - Verify IP address is correct

5. **Memory Issues**
   - Large documents may require more RAM
   - Consider reducing chunk size for very large documents

### Performance Tips
- Use smaller chunk sizes for faster processing
- SQLite handles large documents efficiently
- Chat history is automatically managed
- Google Gemini provides faster responses than Hugging Face

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Google Gemini** for superior AI capabilities
- **Hugging Face** for reliable fallback AI
- **Streamlit** for the beautiful web interface
- **SQLite** for reliable local database storage
- **PyPDF2 & pdfplumber** for PDF text extraction
- **LangChain** for the AI integration framework

## 🆚 AI Model Comparison

| Feature | Google Gemini Pro | Hugging Face | Text Matching |
|---------|------------------|--------------|---------------|
| **Quality** | 🚀 Excellent | 💡 Good | 📝 Basic |
| **Speed** | ⚡ Fast | 🐌 Slower | ⚡ Fast |
| **Cost** | 💰 Free | 💰 Free | 💰 Free |
| **Internet** | ✅ Required | ✅ Required | ❌ Not Required |
| **API Key** | ✅ Required | ✅ Required | ❌ Not Required |
| **Context** | 🧠 Advanced | 🧠 Good | 🧠 Basic |

## 🎯 Why Choose This Project?

- **Best AI Quality** - Google Gemini Pro for superior responses
- **Reliability** - Multiple AI models ensure it always works
- **Cost Effective** - Completely free to use
- **Flexible Deployment** - Works locally and in the cloud
- **Easy Setup** - Simple API key configuration
- **Modern UI** - Beautiful Streamlit interface
- **Document Intelligence** - Advanced PDF processing and Q&A

---

**Perfect for:** Students, researchers, businesses, and anyone who wants a powerful, free document Q&A system with the best AI models available.

**Start your AI-powered document analysis today!** 🚀📚