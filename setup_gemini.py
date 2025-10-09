#!/usr/bin/env python3
"""
Setup script for Google Gemini API key
Run this script to set up your Gemini API key
"""

import os

def setup_gemini_api():
    """Set up Google Gemini API key with Hugging Face fallback."""
    gemini_api_key = "AIzaSyAeXm9GA3NQktwCT8yoaJMrxn3f5YDlMv4"
    huggingface_api_key = "your_huggingface_api_key_here"  # Replace with your actual key
    
    # Set environment variables
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key
    
    print("✅ Google Gemini API key configured (Primary)!")
    print(f"🔑 Gemini API Key: {gemini_api_key[:20]}...")
    print("✅ Hugging Face API key configured (Fallback)!")
    print(f"🔑 Hugging Face API Key: {huggingface_api_key[:20]}...")
    print("🚀 Ready to use Google Gemini Pro with Hugging Face fallback!")
    
    return True

if __name__ == "__main__":
    setup_gemini_api()
