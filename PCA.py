############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# PCA.py
# Apply PCA to datasets for visualization
# Possibly use t-SNE or UMAP for Wine datasets
############################################################

from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

# PCA Computation
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        X = np.array(X)
        
        # Standardize data
        self.mean = np.mean(X, axis=0)
        X_s = X - self.mean
        
        # Covariance
        Sigma = np.cov(X_s, rowvar=False)
        
        # Eigenvector/value problem
        eigenvalues, eigenvectors = np.linalg.eig(Sigma)
        
        # Sort eigenvectors by highest eigenvalue
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]
        
        # Choose top n_component eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Returns dot product of initial dataset (strandardized)
        # and the top eigenvectors
        return np.dot(X_s, self.components)
        
        
# 2D PCA Plot
def pca_2d(X, y, title="2D PCA"):
    X = np.array(X)
    y = np.array(y)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure()
    
    classes = np.unique(y)
    for c in classes:
        plt.scatter(
            X_2d[y == c, 0],
            X_2d[y == c, 1],
            label=f"Class {c}",
            alpha=0.7
        )
    
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# t-SNE 2D
def tsne_2d(X, y, title="t-SNE 2D", perplexity=30):
    X = np.array(X)
    y = np.array(y)
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    plt.figure()
    classes = np.unique(y)
    for c in classes:
        plt.scatter(
            X_2d[y == c, 0],
            X_2d[y == c, 1],
            label=f"Class {c}",
            alpha=0.7
        )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# UMAP 2D
def umap_2d(X, y, title="UMAP 2D", n_neighbors=15):
    X = np.array(X)
    y = np.array(y)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=None)
    X_2d = reducer.fit_transform(X)
    
    plt.figure()
    for c in np.unique(y):
        plt.scatter(X_2d[y == c, 0], X_2d[y == c, 1], label=f"Class {c}", alpha=0.7)
    
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.close()

# umap_2d(X_raisin, y_raisin, "UMAP Raisin 2D", 15)


