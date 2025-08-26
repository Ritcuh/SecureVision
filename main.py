import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point for the SecureVision application."""
    st.set_page_config(
        page_title="SecureVision",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import and run the dashboard
    from app.dashboard import run_dashboard
    run_dashboard()

if __name__ == "__main__":
    main()