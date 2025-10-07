# 🤖 Intellichat - Document Q&A Assistant

Intellichat is an intelligent document Q&A assistant that allows you to upload PDF documents and ask questions about them using natural language. Powered by Microsoft DialoGPT-large and intelligent text matching, it provides accurate, context-aware answers based on your documents with optional web search integration.

## ✨ Features

- 📄 **PDF Document Processing**: Upload and process PDF documents with advanced text extraction
- 🧠 **AI-Powered Q&A**: Ask questions in natural language and get intelligent answers using Microsoft DialoGPT-large
- 💭 **Conversational Memory**: Maintains context across multiple questions in a conversation
- 🔍 **Intelligent Text Search**: Uses smart keyword matching for accurate document retrieval
- 🌐 **Web Search Integration**: Optional web search for enhanced responses
- 🎨 **Beautiful UI**: Modern, responsive interface built with Streamlit
- ⚡ **Real-time Processing**: Fast document processing and instant responses
- 🛡️ **Reliable Operation**: No API failures or rate limits

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Hugging Face API key (FREE)

### Installation

1. **Clone or download the project**
   ```bash
   cd Intellichat
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Get free API key from: https://huggingface.co/settings/tokens
   - Create `.env` file with:
     ```
     HUGGINGFACE_API_KEY=your_huggingface_api_key_here
     ```

5. **Run the application**
   ```bash
   streamlit run app.py --server.port 8502
   ```

6. **Open your browser**
   - Navigate to: http://localhost:8502
   - Start chatting with your documents!

## 📋 Usage

1. **Upload a PDF**: Use the file uploader in the left sidebar to upload your PDF document
2. **Wait for Processing**: The app will extract text, create embeddings, and build a knowledge base
3. **Ask Questions**: Type your questions in natural language
4. **Get Answers**: Receive intelligent, context-aware responses based on your document
5. **Continue Conversations**: Ask follow-up questions that build on previous answers

## 🛠️ Technical Architecture

### Components

- **Frontend**: Streamlit web interface
- **Document Processing**: PyPDF2 and pdfplumber for text extraction
- **Text Chunking**: RecursiveCharacterTextSplitter for optimal chunk sizes
- **AI Model**: Microsoft DialoGPT-large for intelligent responses
- **Database**: SQLite for persistent document storage and retrieval
- **Text Search**: Smart keyword matching with SQLite indexing
- **Web Integration**: DuckDuckGo API for enhanced responses
- **Memory**: SQLite database for reliable operation

### Workflow

1. **Document Upload** → PDF text extraction
2. **Text Processing** → Chunking and SQLite database storage
3. **Query Processing** → SQLite search + AI model generation
4. **Web Enhancement** → Optional web search for additional context
5. **Response Delivery** → Contextual answers with persistent memory

## 🔧 Configuration

### Environment Variables

- `HUGGINGFACE_API_KEY`: Your Hugging Face API key (required)
- **Note**: Hugging Face API key is FREE and easy to get!

### Customization Options

You can modify the following parameters in `app.py`:

- **Chunk Size**: `chunk_size=1000` (adjust for document complexity)
- **Chunk Overlap**: `chunk_overlap=200` (for better context continuity)
- **Search Results**: `k=5` (number of relevant chunks to retrieve)
- **Model Temperature**: `temperature=0.7` (creativity vs accuracy)
- **Web Search**: Toggle in sidebar for enhanced responses
- **Database**: SQLite file-based storage (no external database required)

## 📁 Project Structure

```
Intellichat/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variables template
├── README.md             # This file
├── intellichat.db        # SQLite database (created automatically)
├── docs/                 # Documentation folder
└── venv/                 # Virtual environment
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your OpenAI API key is correctly set in the `.env` file
   - Check that you have sufficient API credits

2. **PDF Processing Issues**
   - Try with a different PDF file
   - Ensure the PDF contains extractable text (not just images)

3. **Memory Issues**
   - Large documents may require more RAM
   - Consider reducing chunk size for very large documents

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Performance Tips

- Use smaller chunk sizes for faster processing
- Clear chat history periodically to free memory
- Process documents in smaller sections if needed

## 🤝 Contributing

Feel free to contribute to this project by:

1. Reporting bugs
2. Suggesting new features
3. Submitting pull requests
4. Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Microsoft** for providing the DialoGPT-large model
- **Hugging Face** for the model hosting and API
- **LangChain** for the powerful framework
- **Streamlit** for the beautiful web interface
- **SQLite** for reliable database storage
- **DuckDuckGo** for web search capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the Streamlit interface
3. Ensure all dependencies are properly installed
4. Verify your Hugging Face API key is valid and properly configured

---

**Happy Document Chatting! 🚀📚**
