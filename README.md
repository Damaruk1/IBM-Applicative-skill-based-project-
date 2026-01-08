 🌍 Multilingual News Bias Analyzer  
### IBM NLU + Gemini Translation + Streamlit Dashboard

This project analyzes **sentiment bias and emotional tone** in multilingual news articles.  
Users can upload any number of `.txt` files in different languages, and the system will:

- Translate articles to English using **Gemini 2.5 Flash**
- Extract **sentiment** and **emotion** using **IBM Watson NLU**
- Compare results visually using advanced graphs
- Display detailed analysis per article
- Provide an interactive **Streamlit dashboard UI**

---

## ✨ Features

### 🔮 AI / NLP Features  
- Multilingual → English Translation (Gemini AI)  
- Sentiment score extraction (IBM NLU)  
- Emotion analysis (joy, sadness, anger, fear, disgust)  
- Per-article and cross-article comparison  
- Auto bias summary + dominant emotion detection  

### 📊 Visualization Features  
- Sentiment bar chart (Plotly)  
- Emotion heatmap  
- Multi-article emotion radar chart  
- Emotion intensity chart  
- Comparison dashboard  
- Download CSV of results  

### 💻 Frontend (Streamlit)  
- Upload multiple text files  
- Beautiful UI with icons and animations  
- Real-time progress bar  
- Detailed text + translation viewer  
- Dashboard style layout  
- Fully responsive and interactive  

### ⚙ Backend (Python)  
- `code_files.py` handles:  
  - Translation  
  - NLU analysis  
  - Error handling  
  - GPU/CPU compatibility  
- Completely independent from the frontend  
- Clean and modular structure  
