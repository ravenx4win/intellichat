"""
Streamlit Cloud Entry Point for Intellichat
This file ensures the app runs correctly on Streamlit Cloud
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main app
if __name__ == "__main__":
    # Import the main app
    from app import *
    
    # The app will run automatically when this file is executed
    # Streamlit Cloud will handle the rest
