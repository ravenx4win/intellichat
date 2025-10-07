# 🤖 Intellichat - Document Q&A Assistant

Intellichat is an intelligent document Q&A assistant that allows you to upload PDF documents and ask questions about them using natural language. Powered by Hugging Face models and LangChain, it provides accurate, context-aware answers based on your documents.

## ✨ Features

- 📄 **PDF Document Processing**: Upload and process PDF documents with advanced text extraction
- 🧠 **AI-Powered Q&A**: Ask questions in natural language and get intelligent answers
- 💭 **Conversational Memory**: Maintains context across multiple questions in a conversation
- 🔍 **Semantic Search**: Uses vector embeddings for accurate document retrieval
- 🎨 **Beautiful UI**: Modern, responsive interface built with Streamlit
- ⚡ **Real-time Processing**: Fast document processing and instant responses

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
- **Embeddings**: OpenAI embeddings for semantic understanding
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM**: OpenAI GPT-3.5-turbo for generating responses
- **Memory**: ConversationBufferMemory for maintaining context

### Workflow

1. **Document Upload** → PDF text extraction
2. **Text Processing** → Chunking and embedding creation
3. **Vector Storage** → ChromaDB vector database
4. **Query Processing** → Semantic search + LLM generation
5. **Response Delivery** → Contextual answers with conversation memory

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_ORG_ID`: Your OpenAI organization ID (optional)

### Customization Options

You can modify the following parameters in `app.py`:

- **Chunk Size**: `chunk_size=1000` (adjust for document complexity)
- **Chunk Overlap**: `chunk_overlap=200` (for better context continuity)
- **Retrieval Count**: `search_kwargs={"k": 4}` (number of relevant chunks)
- **Model Temperature**: `temperature=0.7` (creativity vs accuracy)

## 📁 Project Structure

```
Intellichat/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── env_template.txt       # Environment variables template
├── README.md             # This file
├── docs/                 # Documentation folder
├── venv/                 # Virtual environment
└── chroma_db/            # Vector database storage (auto-created)
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

- **OpenAI** for providing the GPT models and embeddings
- **LangChain** for the powerful framework
- **Streamlit** for the beautiful web interface
- **ChromaDB** for vector storage capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the Streamlit interface
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid and has sufficient credits

---

**Happy Document Chatting! 🚀📚**
