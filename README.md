# EarcareAI
### README for AI-Powered Ear Health Monitoring System  

# 🎧 AI-Powered Ear Health Monitoring  
An interactive Streamlit application that analyzes hearing health using AI-driven clustering. This tool processes audio files, extracts key features, and classifies them into hearing risk categories.  

## 🚀 Features  
- Upload **.wav** audio files for hearing risk analysis.  
- **Mel Spectrogram & MFCC Extraction** for feature representation.  
- **K-Means Clustering** to classify risk levels (Safe, Moderate Risk, High Risk).  
- **PCA & t-SNE** for dimensionality reduction and visualization.  
- **Silhouette Score Calculation** for clustering evaluation.  
- **Interactive Graphs** using Plotly for easy interpretation.  
- **Real-time Prediction** of hearing health for uploaded audio files.  

## 🛠️ Technologies Used  
- **Streamlit** (Frontend & UI)  
- **Librosa** (Audio Processing)  
- **KMeans** (Clustering Model)  
- **PCA & t-SNE** (Dimensionality Reduction)  
- **Plotly** (Data Visualization)  
- **Scikit-learn** (ML Algorithms)  

## 📦 Installation  
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/your-repo-name/ear-health-monitoring.git
   cd ear-health-monitoring
   ```  
2. **Install Dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  
3. **Run the Application:**  
   ```bash
   streamlit run app.py
   ```  

## 📂 File Structure  
```
/ear-health-monitoring
│── app.py                   # Main Streamlit Application  
│── requirements.txt         # Dependencies  
│── README.md                # Project Documentation  
│── data/                    # Sample audio files (if applicable)  
```  

## 📊 How It Works  
1. Upload **.wav** files.  
2. The system extracts **MFCC & Mel Spectrogram** features.  
3. **K-Means** clusters audio files into **3 risk categories**:  
   - 🟢 Safe  
   - 🟡 Moderate Risk  
   - 🔴 High Risk  
4. **Visualizations** (t-SNE & Histograms) help in understanding the clusters.  
5. **Test a New File** to predict its hearing risk category.  

## 📌 Future Improvements  
- Support for **real-time microphone input analysis**.  
- Fine-tuned **deep learning models** for improved classification.  
- Integration of **medical recommendations** from audiologists.  

## 🤝 Contributing  
Feel free to fork this repository and submit pull requests for enhancements!  

## 📜 License  
This project is licensed under the **MIT License**.  

---  
🚀 **Developed with ❤️ for better hearing health!** 🎧
