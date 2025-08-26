🚀 Running SecureVision: Step-by-Step Guide
Prerequisites
System Requirements
Python 3.8 or higher
4GB RAM minimum (8GB recommended)
2GB free disk space
Internet connection (for initial setup)
Installation & Setup
1. Install Dependencies
bash
# Install required packages
pip install streamlit torch torchvision opencv-python Pillow numpy pandas plotly

# Optional: Install transformers for enhanced NLP (falls back to mock if unavailable)
pip install transformers

# Development/Testing (optional)
pip install pytest pytest-cov
2. Verify Installation
Create a simple test file to verify everything works:

python
# test_installation.py
import streamlit as st
import torch
import cv2
import numpy as np
print("✅ All core dependencies installed successfully!")
Run: python test_installation.py

Running the Application
Method 1: Direct Run (Recommended)
bash
# Navigate to project directory
cd SecureVision

# Run the main application
streamlit run main.py
Method 2: Using Python Module
python
# Alternative if streamlit command doesn't work
python -m streamlit run main.py
Method 3: Custom Port/Host
bash
# Run on specific port
streamlit run main.py --server.port 8502

# Run on specific host (for network access)
streamlit run main.py --server.address 0.0.0.0
🌐 Accessing the Dashboard
Open Browser: Navigate to http://localhost:8501
Wait for Loading: Initial model loading may take 30-60 seconds
Start Analysis: Upload files and begin security analysis
📁 Sample Usage
Test with Provided Sample
Use the sample insecure login form in data/code_snippets/insecure_login_form.txt:

html
<form method="get" action="/login">
  <input type="text" name="username">
  <input type="password" name="password">
  <input type="submit" value="Login">
</form>
Create Your Own Test Files
Vulnerable Code Sample (save as test_vulnerable.html):

html
<script>
    var userInput = document.getElementById('search').value;
    document.write("You searched for: " + userInput);
    
    var query = "SELECT * FROM users WHERE name = '" + userInput + "'";
    executeQuery(query);
</script>

<form method="get" action="/login">
    <input type="password" name="pass">
    <input type="submit" value="Login">
</form>
Screenshot Sample: Take a screenshot of any web form or login page and save as PNG/JPG.

🔧 Troubleshooting Common Issues
Issue 1: "Command 'streamlit' not found"
bash
# Solution: Use python -m streamlit instead
python -m streamlit run main.py

# Or ensure streamlit is in PATH
pip install --upgrade streamlit
Issue 2: "ModuleNotFoundError: No module named 'transformers'"
bash
# This is expected - the app will use mock NLP model
# For full functionality, install transformers:
pip install transformers

# Or continue with mock model (works fine for testing)
Issue 3: "Torch not found" or CUDA issues
bash
# Install CPU-only PyTorch (recommended for most users)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
Issue 4: OpenCV installation issues
bash
# Try alternative OpenCV installation
pip uninstall opencv-python
pip install opencv-python-headless
Issue 5: Memory errors with large images
Resize images to under 2MB before upload
Use PNG format for screenshots
Close other applications to free memory
🎯 Quick Test Workflow
1. Test Code Analysis Only
Go to Analysis tab
Select "Code Only" mode
Choose "Paste Code"
Paste the vulnerable code sample above
Click "Run Security Analysis"
Review detected vulnerabilities
2. Test Screenshot Analysis
Select "Screenshot Only" mode
Upload any web page screenshot
Run analysis
Check UI risk factors detected
3. Test Full Analysis
Select "Full Analysis" mode
Upload both screenshot and code file
Get comprehensive security assessment
📊 Expected Results
For Vulnerable Code Sample:
Detections: XSS, SQL Injection, Insecure Authentication, CSRF
Risk Level: High
OWASP Categories: A03_2021 (Injection), A07_2021 (Auth Failures)
Recommendations: Multiple security improvement suggestions
For Screenshots:
UI Analysis: Form detection, input field counting
Risk Factors: Based on visual patterns
Layout Complexity: Structural analysis metrics
⚙️ Performance Optimization
For Better Performance:
bash
# Set environment variables for faster startup
export STREAMLIT_SERVER_ENABLECORS=false
export STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false

# Run with specific config
streamlit run main.py --server.maxUploadSize 200
Memory Management:
Process files under 10MB
Clear browser cache if UI becomes slow
Restart application if memory usage grows large
🔄 Development Mode
For Development/Testing:
bash
# Run with auto-reload on code changes
streamlit run main.py --server.runOnSave true

# Enable debug mode
streamlit run main.py --logger.level debug
Running Tests:
bash
# Run all tests
python tests/test_nlp.py
python tests/test_vision.py

# Or with pytest if installed
pytest tests/ -v
🎉 Success Indicators
You'll know everything is working when you see:

✅ Streamlit dashboard loads without errors
✅ File upload interfaces are responsive
✅ Analysis completes with results displayed
✅ Charts and visualizations render properly
✅ No error messages in terminal/browser console
📞 Getting Help
If you encounter issues:

Check Terminal Output: Look for error messages and stack traces
Browser Console: Check for JavaScript errors (F12 → Console)
File Formats: Ensure uploaded files are in supported formats
Dependencies: Verify all required packages are installed
Python Version: Ensure you're using Python 3.8+
🚀 Next Steps
Once running successfully:

Explore all dashboard tabs (Analysis, Dashboard, Reports, Settings)
Test with your own web application files
Review the comprehensive README.md for detailed features
Check out the test files for more examples
Customize settings for your specific use case
Happy Security Analysis! 🔒

