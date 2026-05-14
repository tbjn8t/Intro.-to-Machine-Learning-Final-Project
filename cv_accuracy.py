############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# cv_accuracy.py
# Functions for each CV method
# Also repeated functions for more
# effecient program calling
############################################################

from sklearn.preprocessing import StandardScaler
import numpy as np
import time

def holdout_cv(model_class, X, y, test_size=0.2, shuffle=True, random_state=None):
    X = np.array(X)
    y = np.array(y)
    
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        
    split = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = model_class()
    model.fit(X_train, y_train)
    
    return model.score(X_test, y_test)


def kfold_cv(model_class, X, y, k=5, shuffle=True, random_state=None):
    X = np.array(X)
    y = np.array(y)
     
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        
    fold_sizes = np.full(k, len(X) // k)
    fold_sizes[:len(X) % k] += 1
    
    scores = []
    start = 0
    
    for fold_size in fold_sizes:
        end = start + fold_size
        
        X_test = X[start:end]
        y_test = y[start:end]
        
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = model_class()
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
        
        start = end
                   
    return np.mean(scores), np.std(scores)


def loocv(model_class, X, y):
    X = np.array(X)
    y = np.array(y)
    
    scores = []
    
    start_total = time.perf_counter()
    
    for i in range(len(X)):        
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        
        X_train = X[mask]
        y_train = y[mask]
        
        X_test = X[i].reshape(1, -1)
        y_test = y[i]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
         
        model = model_class() 
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test)) 
    
    end_total = time.perf_counter()
    total_time = end_total - start_total
               
    return (
        np.mean(scores),
        np.std(scores),
        total_time / len(X),
        total_time
    )
          
          
def repeated_holdout(model_class, X, y, n_runs=20, test_size=0.2, seed_offset=0):
    times = []
    scores = []
    for i in range(n_runs):
        seed = seed_offset + i
        
        start = time.perf_counter()
        avg_acc = holdout_cv(model_class, X, y, test_size=test_size, shuffle=True, random_state=seed)             
        end = time.perf_counter()
        
        times.append(end - start)
        scores.append(avg_acc)
        
    return (
        np.mean(scores),
        np.std(scores),
        np.mean(times),
        np.sum(times)
    )

def repeated_kfold(model_class, X, y, k=5, n_runs=20, seed_offset=0):
    times = []
    scores = []
    
    for i in range(n_runs):
        seed = seed_offset + i
        
        start = time.perf_counter()
        avg_acc, _ = kfold_cv(model_class, X, y, k=k, shuffle=True, random_state=seed)
        
        end = time.perf_counter()
        
        times.append(end - start)
        scores.append(avg_acc)
        
    return (
        np.mean(scores),
        np.std(scores),
        np.mean(times),
        np.sum(times)
    )

def print_acc_report(mean, std, avg_time, tot_time, title="Accuracy Report"):  
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    
    # Confusion matrix
    print("Accuracy Results:")

    print(f"Mean: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Avg Time: {avg_time:.4f}")
    print(f"Total Time: {tot_time:.4f}")     
    print("=" * 50 + "\n")     
            