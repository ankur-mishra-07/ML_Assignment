# UnComment the following line to install streamlit if not already installed
# import subprocess
# subprocess.check_call(["pip3", "install", "streamlit"])

# %% [markdown]
# 
# We will use the fashion-MNIST dataset for this question (you can download it from any
# other source also including libraries). Flatten and preprocess the data (if required) before starting
# the tasks. It will become a 784-dimensional data with 10 classes, more details are available in the
# link
# 
# # a) Train the k-means model on f-MNIST data with k = 10 and 10 random 784-dimensional points (in input range) as initializations. Report the number of points in each cluster.

# %%
# Install kagglehub package
# pip install kagglehub


# UnComment the following lines to install streamlit if not already installed
import subprocess
subprocess.check_call(["pip", "install", "streamlit"])
subprocess.check_call(["pip", "install", "scikit-learn"])
subprocess.check_call(["pip", "install", "matplotlib"])
subprocess.check_call(["pip", "install", "kagglehub"])

import os
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import kagglehub  # type: ignore

# Initialize session state variables if not already present
if 'X_Train' not in st.session_state:
    st.session_state.X_Train = None
if 'X_Test' not in st.session_state:
    st.session_state.X_Test = None
if 'y_Train' not in st.session_state:
    st.session_state.y_Train = None
if 'y_Test' not in st.session_state:
    st.session_state.y_Test = None
if 'kmeans' not in st.session_state:
    st.session_state.kmeans = None
if 'kmeans2' not in st.session_state:
    st.session_state.kmeans2 = None
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = None

# an output log container.
log_container = st.container()
def append_output(message: str):
    """Append a message to the persistent log and update the display."""
    st.session_state.log += message + "\n"
    with log_container:
        st.text_area("Output Log", st.session_state.log, height=300, key="output_log")


# Sliders for user input
no_of_clusters = int(st.slider("Select number of clusters ", 1, 10, 10))
no_of_images_per_cluster = int(st.slider("Select number of images per cluster ", 1, 10, 10))
st.write("Number of clusters selected:", no_of_clusters)
st.write("Number of images per cluster selected:", no_of_images_per_cluster)

def load_idxfile_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip magic number and dimension info (16 bytes)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def load_idxfile_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # Skip magic number and dimension info (8 bytes)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def load_fashion_mnist(path, isInformedCluster):
    st.write("Loading and reading the Fashion-MNIST data...")
    
    # Determine full file paths
    train_images_path = os.path.join(path, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels-idx1-ubyte')
    test_images_path  = os.path.join(path, 't10k-images-idx3-ubyte')
    test_labels_path  = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    # Load raw files
    train_images_raw = load_idxfile_images(train_images_path)
    train_labels = load_idxfile_labels(train_labels_path)
    test_images_raw = load_idxfile_images(test_images_path)
    test_labels = load_idxfile_labels(test_labels_path)
    
    read_fashion_mnist(train_images_raw, train_labels, test_images_raw, test_labels, isInformedCluster)

def read_fashion_mnist(train_images_raw, train_labels, test_images_raw, test_labels, isInformedCluster):
    st.write("Reshaping and normalizing data...")
    
    train_images_raw = train_images_raw.reshape(-1, 28, 28)
    test_images_raw = test_images_raw.reshape(-1, 28, 28)

    # Update session_state with flattened and normalized images.
    st.session_state.X_Train = train_images_raw.reshape(-1, 784).astype(np.float32) / 255.0
    st.session_state.X_Test = test_images_raw.reshape(-1, 784).astype(np.float32) / 255.0
    st.session_state.y_Train = train_labels
    st.session_state.y_Test = test_labels

    st.write("Training data shape:", st.session_state.X_Train.shape)
    st.write("Test data shape:", st.session_state.X_Test.shape)
    
    # If not using informed clustering, proceed with training the first model.
    train_kmeans_model()

def train_kmeans_model():
    st.write("Training the K-Means model with random initialization (k = 10)...")
    
    # Generate 10 random initialization points in the valid [0, 1] range
    random_init_points = np.random.rand(no_of_clusters, 784).astype(np.float32)

    # Create and fit the k-means model
    st.session_state.kmeans = KMeans(
        n_clusters=no_of_clusters,
        init=random_init_points,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    st.session_state.kmeans.fit(st.session_state.X_Train)
    
    report_cluster_counts()

def report_cluster_counts():
    st.write("Reporting the number of points in each cluster:")
    labels = st.session_state.kmeans.labels_
    counts = np.bincount(labels)
    
    for i in range(no_of_clusters):
        st.write(f"Cluster {i}: {counts[i]} points")
        print(f"Cluster {i}: {counts[i]} points")
        
    visualize_cluster_centers()
    visualize_cluster_images()

def visualize_cluster_centers():
    st.write("Visualizing the cluster centers as 28x28 images...")
    cluster_centers = st.session_state.kmeans.cluster_centers_
    
    fig, axes = plt.subplots(1, no_of_clusters, figsize=(15, 4))
    for i, ax in enumerate(axes):
        center_image = cluster_centers[i].reshape(28, 28)
        ax.imshow(center_image, cmap='gray')
        ax.set_title(f"Cluster {i}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def visualize_cluster_images():
    st.write("Visualizing sample images from each cluster...")
    cluster_labels = st.session_state.kmeans.labels_
    fig, axes = plt.subplots(no_of_clusters, no_of_images_per_cluster, figsize=(15, 15))
    
    for cluster in range(no_of_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        # Take the first available images for the cluster (adjust as needed)
        selected_indices = cluster_indices[:no_of_images_per_cluster]
        
        for i, idx in enumerate(selected_indices):
            img = st.session_state.X_Train[idx].reshape(28, 28)
            ax = axes[cluster, i]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(f"Cluster {cluster}", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def load_fashion_mnist2():
    st.write("Training K-Means with informed initialization (10 images from each class)...")
    # Check that data is already loaded
    if st.session_state.X_Train is None or st.session_state.y_Train is None:
        st.error("Please download the dataset first.")
        return

    # Compute initial centers: average of 10 random images for each class 0-9.
    init_centers = []
    for label in range(10):
        indices = np.where(st.session_state.y_Train == label)[0]
        sampled_indices = np.random.choice(indices, size=10, replace=False)
        avg_image = np.mean(st.session_state.X_Train[sampled_indices], axis=0)
        init_centers.append(avg_image)
    init_centers = np.array(init_centers)

    # Train k-means model with informed initialization
    st.session_state.kmeans2 = KMeans(
        n_clusters=10,
        init=init_centers,
        n_init=1,  # Use the provided initialization only
        max_iter=300,
        random_state=42
    )
    st.session_state.kmeans2.fit(st.session_state.X_Train)
    
    # Report cluster counts for informed initialization
    st.write("Number of points in each informed cluster:")
    cluster_labels = st.session_state.kmeans2.labels_
    counts = np.bincount(cluster_labels)
    for i in range(10):
        st.write(f"Cluster {i}: {counts[i]} points")
        print(f"Cluster {i}: {counts[i]} points")
    
    # Visualize cluster centers and sample images
    visualize_informed_cluster_centers()
    visualize_informed_cluster_images()

def visualize_informed_cluster_centers():
    st.write("Visualizing informed cluster centers as 28x28 images...")
    cluster_centers = st.session_state.kmeans2.cluster_centers_
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 4))
    for i, ax in enumerate(axes):
        center_image = cluster_centers[i].reshape(28, 28)
        ax.imshow(center_image, cmap='gray')
        ax.set_title(f"Cluster {i}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def visualize_informed_cluster_images():
    st.write("Visualizing sample images from informed clusters...")
    cluster_labels = st.session_state.kmeans2.labels_
    fig, axes = plt.subplots(10, no_of_images_per_cluster, figsize=(15, 15))
    
    for cluster in range(10):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        selected_indices = cluster_indices[:no_of_images_per_cluster]
        
        for i, idx in enumerate(selected_indices):
            img = st.session_state.X_Train[idx].reshape(28, 28)
            ax = axes[cluster, i]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(f"Cluster {cluster}", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def evaluate_sse_kmeans():
    st.write("Evaluating clustering quality using SSE (Sum of Squared Errors)...")
    if st.session_state.kmeans is None or st.session_state.kmeans2 is None:
        st.error("Both clustering models must be trained. Ensure that you have run both clustering methods.")
        return
    sse_kmeans = st.session_state.kmeans.inertia_
    sse_kmeans2 = st.session_state.kmeans2.inertia_
    
    st.write("SSE for random initialization (kmeans):", sse_kmeans)
    st.write("SSE for informed initialization (kmeans2):", sse_kmeans2)
    print("SSE for random init:", sse_kmeans)
    print("SSE for informed init:", sse_kmeans2)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Random Init', 'Informed Init']
    sse_values = [sse_kmeans, sse_kmeans2]
    ax.bar(models, sse_values, color=['blue', 'orange'])
    ax.set_ylabel('SSE (Inertia)')
    ax.set_title('SSE Comparison')
    ax.set_yscale('log')
    ax.grid(axis='y')
    st.pyplot(fig)
    plt.close()

def downLoad_dataset():
    st.title("K-Means Clustering on Fashion-MNIST Dataset")
    st.write("Downloading the Fashion-MNIST dataset using kagglehub...")
    st.session_state.dataset_path = kagglehub.dataset_download("zalando-research/fashionmnist")
    st.write("Dataset downloaded. Path:", st.session_state.dataset_path)
    load_fashion_mnist(st.session_state.dataset_path, False)

# Layout buttons for user interactions.
st.button("Download dataset and Train Random Cluster", key="download_dataset", on_click=downLoad_dataset)
st.button("Train Informed Cluster", key="train_informed", on_click=load_fashion_mnist2)
st.button("Evaluate SSE", key="evaluate_sse", on_click=evaluate_sse_kmeans)
