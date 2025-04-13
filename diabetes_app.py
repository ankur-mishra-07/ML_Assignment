# UnComment the following lines to install streamlit if not already installed
import subprocess
subprocess.check_call(["pip3", "install", "streamlit"])
subprocess.check_call(["pip3", "install", "scikit-learn"])
subprocess.check_call(["pip3", "install", "matplotlib"])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


accuracy_over_PCA = 0
accuracy_reduced_PCA = 0
opt_components = 0
X_scaled = 0
y = 0


global option
option = st.selectbox(
    "Select the model you want to use",
    ("Logistic Regression", "Random Forest"),
    index=None,
    placeholder="Select the model you want to use...",
)

st.write("You selected:", option)

def loadDiabetesDataset():
    # Loading the dataset from google Drive
    df = pd.read_csv('diabetes.csv')

    st.write("Data Head", df.describe())

    global y
    # Data Preparation :- through which we can check null and features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    global X_scaled
    X_scaled = scaler.fit_transform(X)

    # Apply PCA and get explained variance ratio
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    global opt_components
    opt_components = np.argmax(explained_variance >= 0.95) + 1
    # a) Find the optimum number of principal components for the features in the above-mentioned data.

    pca =PCA()
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Prinicipal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Optimum_PCA')
    plt.grid(True)
    plt.show()
    st.pyplot(plt)

    # b) Plot the explained variance ratio for each principal component.
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio of Principal Components')
    plt.grid(True)
    st.pyplot(plt)
    # c) Plot the cumulative explained variance ratio.
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Ratio')
    plt.grid(True)
    st.pyplot(plt)
    # d) Plot the PCA projection of the data.
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    plt.title('PCA Projection of the Data') 
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Diabetes Progression')
    plt.grid(True)
    st.pyplot(plt)
    # e) Train a model using the PCA-transformed data.
    # f) Evaluate the model performance.
    # g) Display the model performance metrics.
    # h) Save the model to a file.
    # i) Load the model from the file.
    # j) Make predictions using the loaded model.   

    # Splitting the data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    global option
    trainModel(option, X_train, y_train)

def trainModel(model_name, X_train, y_train):
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    evaluateModel(model, X_train, y_train)

def evaluateModel(model, X_test, y_test):
    y_pred = model.predict(X_test)
    global accuracy_over_PCA
    accuracy_over_PCA = accuracy_score(y_test, y_pred)
    st.write("Model Accuracy with PCA:", accuracy_over_PCA)
    global X_scaled
    accuracy_over_PCA_method(model, X_scaled)

def accuracy_over_PCA_method(model, X_scaled):
    # Check accuracy with PCA reduce data

    pca = PCA(n_components= 6)
    X_pca_reduced = pca.fit_transform(X_scaled)

    global y
    X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.2, random_state=42)
    # Train the model using PCA-transformed data
    model.fit(X_train_reduced, y_train)

    global accuracy_reduced_PCA, accuracy_over_PCA
    # Evaluate the model performance
    accuracy_reduced_PCA = model.score(X_train_reduced, y_train)
    st.write("Model Accuracy with reduced PCA:", accuracy_reduced_PCA)
    st.write("Accuracy Drop:", accuracy_over_PCA - accuracy_reduced_PCA)

st.button("Generate Result", key="generate_result", on_click=loadDiabetesDataset)
