# This is the main entry point for Streamlit Cloud
# It imports and runs your app.py file

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run your main app
if __name__ == "__main__":
    # Import the main app
    from app import *
    
    # The app will run automatically when imported
