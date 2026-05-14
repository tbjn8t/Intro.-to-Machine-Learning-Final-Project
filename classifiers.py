############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# classifiers.py
# File defines all classifer functions
# Naive Bayes, KNN, PW, LDA, and QDA
############################################################

import numpy as np
from collections import Counter    

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
    
    # Collects class mean, var, prior
    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
             X_c = X[y == c]
             self.mean[c] = np.mean(X_c, axis=0)
             self.var[c] = np.var(X_c, axis=0)
             self.priors[c] = X_c.shape[0] / X.shape[0]
    
    # Finds P(x_i | c)
    # numerator : exp( - (x - μ)^2 / (2σ^2))
    # denominator : sqrt(2πσ^2)
    def gaussian(self, c, x):
        mean = self.mean[c]
        var = self.var[c] + 1e-9
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    # Predicts class for one sample
    # Sums all P(x_i | c) to get P(x | c)
    # logP(c) + logP(x | c) which is proportional to the posterior
    # Choose class with highest posterior value
    def predict_single(self, x):
        posteriors = []
        
        for c in self.classes:
            prior_log = np.log(self.priors[c])
            conditional_log = np.sum(np.log(self.gaussian(c, x) + 1e-9))
            posterior = prior_log + conditional_log
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
    
    # Predict class for samples in X
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
    
    # Give accuracy score
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)    
    
    
class K_Nearest_Neighbor:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y     
         
    def predict_single(self, x):
        # Find distance to training samples
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # Sort distances and get class labels
        # from k nearest neightbors
        k_idx = np.argsort(distances)[:self.k]
        k_nearest = [self.y_train[i] for i in k_idx]
        
        # Return most common class
        return Counter(k_nearest).most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    # Give accuracy score
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)   
    

class ParzenGaussian:
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.classes = None
        self.class_data = {}
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)        
        self.classes = np.unique(y)   
        self.class_data = {}     

        for c in self.classes:
            self.class_data[c] = X[y == c]
    
    def parzen_gaussian(self, x, X_i):
        d = x.shape[0]
        sigma = self.sigma
        diff = X_i - x
        dist2 = np.sum(diff**2, axis=1)
        
        const = 1 / ((2 * np.pi * sigma**2) ** (d/2))

        return np.mean(const * np.exp(-dist2 / (2 * sigma**2)))
    
    def predict_single(self, x):
        scores = []
        n_samples = sum(len(i) for i in self.class_data.values())
        
        for c in self.classes:
            X_i = self.class_data[c]
            density = self.parzen_gaussian(x, X_i)
            prior = len(X_i) / n_samples
            scores.append(np.log(density + 1e-9) + np.log(prior))
        return self.classes[np.argmax(scores)]
    
    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)    


class LDA:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
        
        self.global_mean = None
        
        self.Sw = None
        self.Sb = None
        
        self.eigenvalues = None
        self.eigenvectors = None
        
        self.W = None
        self.projected_means = {}
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Overall mean
        self.global_mean = np.mean(X, axis=0)
        
        # Initialize scatter matrices
        self.Sw = np.zeros((n_features, n_features))
        self.Sb = np.zeros((n_features, n_features))        
        
        # Compute means and priors
        for c in self.classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.means[c] = mean_c
            self.priors[c] = len(X_c) / n_samples
            
            # Sw
            centered = X_c - mean_c
            self.Sw += centered.T @ centered
            
            # Sb
            n_c = X_c.shape[0]
            
            mean_diff = (mean_c - self.global_mean).reshape(n_features, 1)
            self.Sb += n_c * (mean_diff @ mean_diff.T)
        
        # Sw regularization    
        self.Sw += 1e-9 * np.eye(n_features)
        
        # Solve eigenvalue problem
        # Sw^-1 * Sb
        A = np.linalg.inv(self.Sw) @ self.Sb
        
        self.eigenvalues, self.eigenvectors = np.linalg.eig(A)
        
        # Sort eigenvectors by largest eigenvalue
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]
        
        # Kepp top eigenvectors
        n_components = min(n_classes - 1, n_features)
        self.W = np.real(self.eigenvectors[:, :n_components])
        
        # Project class means
        for c in self.classes:
            self.projected_means[c] = self.means[c] @ self.W  
         
    def transform(self, X):
        X = np.array(X)
        return np.real(X @ self.W)
    
    def predict_single(self, x):
        x = np.array(x)
        
        x_proj = np.real(x @ self.W)
        
        distances = []
        
        for c in self.classes:
            mean_proj = self.projected_means[c]
            
            # Euclidean distance
            dist = np.linalg.norm(x_proj - mean_proj)
            distances.append(dist)
            
        return self.classes[np.argmin(distances)]
    
    def predict(self, X):
        X = np.array(X)
        
        predictions = []
        
        for x in X:
            predictions.append(self.predict_single(x))
            
        return np.array(predictions)

    # Give accuracy score
    def score(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        predictions = self.predict(X)        
        
        return np.mean(predictions == y) 
    
      
class QDA:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
        self.covs = {}
        self.cov_invs = {}
        self.cov_dets = {}
    
    def fit(self, X, y):        
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Compute means, priors, and cov
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.priors[c] = len(X_c) / n_samples
            
            cov = np.cov(X_c, rowvar=False)
            
            # Regularization for stability
            cov += 1e-9 * np.eye(n_features)
            
            self.covs[c] = cov
            self.cov_invs[c] = np.linalg.inv(cov)
            self.cov_dets[c] = np.linalg.det(cov)  
         
    def discriminant(self, x, c):
        mean = self.means[c]
        cov_inv = self.cov_invs[c]
        cov_det = self.cov_dets[c]
        
        diff = x - mean
        
        # -(0.5) * log|Sigma| - 0.5 * (x - u)^T * Sigma^-1 * (x - u) + log(P(c))
        term1 = -0.5 * np.log(cov_det)
        term2 = -0.5 * diff @ cov_inv @ diff
        term3 = np.log(self.priors[c])
        
        return term1 + term2 + term3
    
    def predict(self, X):
        X = np.array(X)
        
        predictions = []
        for x in X:
            scores = [self.discriminant(x, c) for c in self.classes]
            predictions.append(self.classes[np.argmax(scores)])
        return np.array(predictions)

    # Give accuracy score
    def score(self, X, y):
        return np.mean(self.predict(X) == y) 