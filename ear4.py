import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Fix font issues for emojis
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Define dataset path
AUDIO_FOLDER = "C:\\Users\\HP\\Downloads\\blind_testset"  # Update this path
NUM_CLUSTERS = 3  # Number of clusters

# Mapping cluster numbers to labels and emojis
CLUSTER_LABELS = {
    0: ("Safe", "🟢"),
    1: ("Moderate Risk", "🟡"),
    2: ("High Risk", "🔴")
}

# Function to extract Mel spectrogram and MFCC features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)

        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Flatten features into a 1D vector
        feature_vector = np.hstack([np.mean(mel_spec_db, axis=1), np.mean(mfccs, axis=1)])
        return feature_vector
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Extract features from all audio files
X = []
file_names = []

for file in os.listdir(AUDIO_FOLDER):
    if file.endswith(".wav"):
        file_path = os.path.join(AUDIO_FOLDER, file)
        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            file_names.append(file)

# Convert to NumPy array
X = np.array(X, dtype=np.float64)  # Ensure dtype is float64

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Evaluate clustering performance using silhouette score
silhouette_avg = silhouette_score(X_pca, clusters)
print(f"\n🔹 **Silhouette Score**: {silhouette_avg:.4f}")

# **Clustering Model Summary**
print("\n📌 **K-Means Clustering Model Summary** 📌\n")
print(f"➡️ Number of Clusters: {NUM_CLUSTERS}")
print(f"➡️ Number of Features After PCA: {X_pca.shape[1]}")
print(f"➡️ Number of Audio Samples: {X_pca.shape[0]}")
print("\n🔹 **Cluster Centers (Reduced Dimensions):**")
print(kmeans.cluster_centers_)

# Visualize clusters using t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plot Clusters (t-SNE Projection)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=[CLUSTER_LABELS[c][0] for c in clusters], palette="viridis", alpha=0.8)
plt.title("🎧 Audio Clusters (t-SNE Projection)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Cluster")
plt.show()

# Plot Cluster Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=[CLUSTER_LABELS[c][0] for c in clusters], hue=[CLUSTER_LABELS[c][0] for c in clusters], palette="viridis", legend=False)
plt.title("📊 Cluster Distribution of Audio Files")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()

# Assign cluster labels to file names
cluster_labels = {file_names[i]: CLUSTER_LABELS[clusters[i]] for i in range(len(file_names))}

# Function to predict the cluster of a given audio file
def predict_cluster(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found → {file_path}")
        return None

    features = extract_features(file_path)
    if features is None:
        return None

    # Convert to float64 to match KMeans requirements
    features_scaled = scaler.transform([features]).astype(np.float64)
    features_pca = pca.transform(features_scaled).astype(np.float64)

    predicted_cluster = kmeans.predict(features_pca)[0]
    return CLUSTER_LABELS[predicted_cluster]

# Select 10 random audio files from the training dataset for testing
random_audio_files = random.sample(file_names, min(10, len(file_names)))

# Predict clusters for 10 iterations
print("\n🔍 **Predicting Hearing Health Status for 10 Audio Files:**\n")
for i, test_audio in enumerate(random_audio_files, 1):
    test_audio_path = os.path.join(AUDIO_FOLDER, test_audio)
    predicted_label = predict_cluster(test_audio_path)
    
    if predicted_label is not None:
        label_text, emoji = predicted_label
        print(f"🎵 **Iteration {i}:** {test_audio} → **Predicted Status:** {label_text} {emoji}")
    else:
        print(f"⚠️ **Iteration {i}:** Failed to extract features from {test_audio}")

# Test on a new audio file
test_audio = "C:\\Users\\HP\\Downloads\\blind_testset\\00723.wav"  # Update with actual test file
predicted_label = predict_cluster(test_audio)

if predicted_label is not None:
    label_text, emoji = predicted_label
    print(f"\n🔊 **Predicted Hearing Health Status for Test File:** {label_text} {emoji}")
else:
    print("❌ Failed to extract features from test file.")
