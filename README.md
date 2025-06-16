# SPEECH RECOGNIZATION

🔊 AUDIO CLASSIFIER APP – USER INSTRUCTIONS

This application classifies audio files into two categories:
- "Conversation"
- "Background"

👣 STEP-BY-STEP SETUP GUIDE

📌 1. Install Python (only if not already installed)
   - Go to: https://www.python.org/downloads/
   - Download the latest Python version (e.g., Python 3.11+)
   - During installation:
       ✅ Check the box: “Add Python to PATH”
       📦 Click “Install Now”

📌 2. Install Required Libraries
   - Open Command Prompt (press Windows + R, type `cmd`, press Enter)
   - Navigate to this folder where the app is located using:
       cd path\to\AudioClassifierAppFolder
       (Example: cd D:\AudioDetection)

   - Run the following command to install all necessary libraries:

       pip install -r requirements.txt

   This will install:
   - `streamlit`
   - `librosa`
   - `numpy`
   - `scikit-learn`
   - `joblib`
   - `matplotlib`

📌 3. Run the App
   - Double-click `launch_app.bat`
   - This will open your browser automatically with the app

📌 4. Test Your Audio
   - Click "Browse files" to upload a `.wav` audio file
   - The app will analyze and display whether it's:
       ✅ "Conversation"
       ✅ "Background"

📌 5. Troubleshooting
   - If the browser doesn't open:
       Open Command Prompt and type:
       streamlit run app.py

   - If you get an error like “module not found”, rerun:
       pip install -r requirements.txt

📩 For help, contact the developer.

---

🖥️ REQUIREMENTS SUMMARY:
- Windows PC
- Python 3.10 or 3.11
- Internet Browser (Chrome, Edge, etc.)

---

✅ Enjoy using the Audio Classifier App!
