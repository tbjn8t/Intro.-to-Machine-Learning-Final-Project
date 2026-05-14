############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# cv_confusion.py
# Create confusion matrices when each CV method is called
############################################################

from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red
from classifiers import NaiveBayes, K_Nearest_Neighbor, LDA, QDA

from sklearn.preprocessing import StandardScaler
import numpy as np

def holdout_cm(model_class, X, y, test_size=0.2, shuffle=True, random_state=0):
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
    
    y_pred = model.predict(X_test)
    
    return y_test, y_pred

def kfold_cm(model_class, X, y, k, shuffle=True, random_state=0):
    X = np.array(X)
    y = np.array(y)
     
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        
    fold_sizes = np.full(k, len(X) // k)
    fold_sizes[:len(X) % k] += 1
    
    y_true = []
    y_pred = []
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

        y_pred_single = model.predict(X_test)
        y_true.append(y_test)
        y_pred.append(y_pred_single)
        
        start = end              
    return np.concatenate(y_true), np.concatenate(y_pred)

def loocv_cm(model_class, X, y):
    X = np.array(X)
    y = np.array(y)
    
    y_true = []
    y_pred = []
    
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
        y_pred_single = model.predict(X_test)[0]
        
        y_true.append(y_test)
        y_pred.append(y_pred_single)
    return np.array(y_true), np.array(y_pred)


def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    class_to_index = {c: i for i, c in enumerate(classes)}
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        i = class_to_index[t]
        j = class_to_index[p]
        cm[i, j] += 1
    return cm, classes

def cm_metrics(cm):
    n_classes = cm.shape[0]
    
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    for i in range(n_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        
        precision[i] = TP / (TP + FP + 1e-9)
        recall[i] = TP / (TP + FN + 1e-9)
    return precision, recall

def print_cm_report(cm, classes=None, title="Confusion Matrix Report"):
    cm = np.array(cm)
    precision, recall = cm_metrics(cm)
    
    n = cm.shape[0]
    
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    if classes is None:
        classes = [f"C{i}" for i in range(n)]
    print("      " + "  ".join(f"{c:>6}" for c in classes))
    
    for i in range(n):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(n))
        print(f"{classes[i]:>5} {row}")
        
    # Precision and Recall
    print("\n")
    print(f"{'Class':>6} {'Precision':>12} {'Recall':>12}")
    print("-" * 35)
    
    for i in range(n):
        print(f"{classes[i]:>6} {precision[i]:12.3f} {recall[i]:12.3f}")       
    print("\n" + "=" * 50 + "\n")
    
#y_test, y_pred = holdout_cm(NaiveBayes, X_raisin, y_raisin, 0.2, True, 42)
#cm, classes = confusion_matrix(y_test, y_pred)
#print_cm_report(cm, classes, title="Raisin Dataset - Naive Bayes - Holdout")

#y_test, y_pred = kfold_cm(NaiveBayes, X_raisin, y_raisin, 5, True, 42)
#cm, classes = confusion_matrix(y_test, y_pred)
#print_cm_report(cm, classes, title="Raisin Dataset - Naive Bayes - Kfold=5")