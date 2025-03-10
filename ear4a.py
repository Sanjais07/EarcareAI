import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Custom Streamlit Styling
st.set_page_config(page_title="🎧 Ear Health Clustering", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #F5F7FA; }
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/786/786432.png", width=120)
st.sidebar.title("🎧 EarCare AI")
st.sidebar.write("AI-Powered Hearing Health Monitoring")
st.sidebar.info("Upload audio files to analyze hearing risk categories.")

# Upload audio files
uploaded_files = st.file_uploader("📎 Upload audio files (.wav)", accept_multiple_files=True, type=["wav"])

# Define number of clusters
NUM_CLUSTERS = 3

# Cluster labels and recommended headphone usage limits with detailed medical risks
CLUSTER_LABELS = {
    0: ("Safe", "🟢", "You can use headphones up to **6 hours per day** without major risk.", "Prolonged use beyond **8 hours** may cause mild ear fatigue."),
    1: ("Moderate Risk", "🟡", "Recommended **2-3 hours per day** to avoid damage.", "Using more than **4 hours** may cause **temporary threshold shifts (TTS)**, leading to muffled hearing and potential **tinnitus (ringing in ears)**."),
    2: ("High Risk", "🔴", "Limit headphone use to **less than 1 hour per day**.", "Exceeding **1-2 hours** could lead to **permanent threshold shifts (PTS)**, **noise-induced hearing loss (NIHL)**, **hyperacusis (increased sensitivity to sound)**, and long-term **auditory nerve damage**.")
}

# Function to extract Mel spectrogram and MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feature_vector = np.hstack([np.mean(mel_spec_db, axis=1), np.mean(mfccs, axis=1)])
        return feature_vector
    except Exception as e:
        st.error(f"⚠️ Error processing {file_path}: {e}")
        return None

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
    
    X = []
    file_names = []
    
    for uploaded_file in uploaded_files:
        file_path = uploaded_file.name
        features = extract_features(uploaded_file)
        
        if features is not None:
            X.append(features)
            file_names.append(file_path)

    if len(X) > 0:
        X = np.array(X, dtype=np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X_scaled)
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)
        silhouette_avg = silhouette_score(X_pca, clusters)
        st.subheader(f"🔹 **Silhouette Score:** {silhouette_avg:.4f}")
        tsne = TSNE(n_components=2, perplexity=10, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        cluster_labels = [CLUSTER_LABELS[c][0] for c in clusters]
        fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=cluster_labels,
                         title="🎧 Audio Clusters (t-SNE Projection)",
                         labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"},
                         color_discrete_map={"Safe": "green", "Moderate Risk": "yellow", "High Risk": "red"})
        st.plotly_chart(fig)
        
        fig = px.histogram(x=cluster_labels, title="📊 Cluster Distribution of Audio Files", color=cluster_labels,
                           color_discrete_map={"Safe": "green", "Moderate Risk": "yellow", "High Risk": "red"})
        st.plotly_chart(fig)
        
        st.subheader("🎶 Uploaded Files & Predicted Categories (Showing up to 20)")
        for i, file_name in enumerate(file_names[:20]):
            label_text, emoji, _, _ = CLUSTER_LABELS[clusters[i]]
            st.write(f"🎵 **{file_name}** → {label_text} {emoji}")
            st.audio(uploaded_files[i], format="audio/wav")

        st.subheader("🔍 Predict Hearing Health for a New Audio File")
        test_file = st.file_uploader("Upload a test audio file (.wav)", type=["wav"], key="test")
        
        if test_file:
            test_features = extract_features(test_file)
            if test_features is not None:
                test_features_scaled = scaler.transform([test_features]).astype(np.float64)
                test_features_pca = pca.transform(test_features_scaled).astype(np.float64)
                predicted_cluster = kmeans.predict(test_features_pca)[0]
                label_text, emoji, usage_time, warning = CLUSTER_LABELS[predicted_cluster]
                st.success(f"🔊 **Predicted Hearing Health Status:** {label_text} {emoji}")
                st.audio(test_file, format="audio/wav")
                st.markdown(f"""
                    **🎧 Recommended Headphone Usage:**  
                    {usage_time}  
                    
                    **⚠️ Medical Warning:**  
                    {warning}
                """)
else:
    st.warning("⚠️ Please upload audio files to start clustering.")
