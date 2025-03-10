import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Define dataset path
AUDIO_FOLDER = "C:\\Users\\HP\\Downloads\\blind_testset"  # Update this path
CSV_FILE_PATH = "C:\\Users\\HP\\Downloads\\ear_data.csv"  # Update to actual file path
NUM_CLUSTERS = 3  # Number of clusters

# Load dataset
df = pd.read_csv(CSV_FILE_PATH)

# Strip column names to remove extra spaces
df.columns = df.columns.str.strip()

# Print column names for debugging
print("Columns in dataset:", df.columns.tolist())

# Convert categorical columns to numerical
le = LabelEncoder()
for col in ['Sex', 'Age_Group', 'Tinnitus_Symptoms', 'Hearing_Loss']:
    if col in df.columns:  # Check if column exists before encoding
        df[col] = le.fit_transform(df[col])
    else:
        print(f"Warning: Column {col} not found in dataset.")

# Standardize numeric features
scaler = StandardScaler()
numeric_cols = ['Listening_Hours', 'Volume_Level']
df_scaled = scaler.fit_transform(df[numeric_cols])

# Extract features from all audio files
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feature_vector = np.hstack([np.mean(mel_spec_db, axis=1), np.mean(mfccs, axis=1)])
        return feature_vector
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

X_audio = []
file_names = []

for file in os.listdir(AUDIO_FOLDER):
    if file.endswith(".wav"):
        file_path = os.path.join(AUDIO_FOLDER, file)
        features = extract_features(file_path)

        if features is not None:
            X_audio.append(features)
            file_names.append(file)

# Convert to NumPy array
X_audio = np.array(X_audio, dtype=np.float64) if X_audio else np.array([])

# Reduce dimensions using PCA for audio data
pca = PCA(n_components=10)
X_audio_pca = pca.fit_transform(X_audio) if X_audio.size else np.array([])

# Combine scaled CSV data and audio features
if X_audio_pca.size > 0:
    X_combined = np.hstack((df_scaled, X_audio_pca[:len(df_scaled)]))
else:
    X_combined = df_scaled

# Clustering
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_combined)

# Evaluate clustering performance
silhouette_avg = silhouette_score(X_combined, clusters)
print(f"\n🔹 **Silhouette Score**: {silhouette_avg:.4f}")

# **Clustering Model Summary**
print("\n📌 **K-Means Clustering Model Summary** 📌\n")
print(f"➡️ Number of Clusters: {NUM_CLUSTERS}")
print(f"➡️ Features Used: {X_combined.shape[1]}")
print(f"➡️ Data Points: {X_combined.shape[0]}")

# Visualize clusters using t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_combined)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette="viridis", alpha=0.8)
plt.title("🎧 Hearing Health Clusters (t-SNE Projection)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.show()
