############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# PCA.py
# Apply PCA to datasets for visualization
# Possibly use t-SNE or UMAP for Wine datasets
############################################################

from cv_accuracy import repeated_holdout, repeated_kfold, loocv, holdout_cv, kfold_cv, print_acc_report
from classifiers import NaiveBayes, K_Nearest_Neighbor, ParzenGaussian, LDA, QDA
from cv_confusion import holdout_cm, kfold_cm, loocv_cm
from cv_confusion import confusion_matrix, print_cm_report
from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools

def criteria_J(X, y):
    X = np.array(X)
    y = np.array(y)
    
    # Number of samples, features, and classes
    n_features = X.shape[1]
    classes = np.unique(y)
    
    # Global mean
    mu_global = np.mean(X, axis=0)
    
    # Initialize Sw and Sb
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))
    
    for c in classes:
        X_c = X[y == c]
        mu_c = np.mean(X_c, axis=0)
        N_c = X_c.shape[0]
        
        # Sw computation
        centered = X_c - mu_c
        Sw += centered.T @ centered
        
        # Sb computation
        diff = (mu_c - mu_global).reshape(-1, 1)
        Sb += N_c * (diff @ diff.T)
    
    # Regularization
    Sw += 1e-6 * np.eye(n_features)
        
    # J1 computation
    J1 = np.trace(np.linalg.inv(Sw) @ Sb)
    
    return J1

def feature_search(X, y, max_features=None, top_k=50):
    X = np.array(X)
    n_features = X.shape[1]
    
    feature_idx = list(range(n_features))
    results = []
    
    for i in range(1, n_features + 1):
        if max_features is not None and i > max_features:
            break
        for subset in itertools.combinations(feature_idx, i):
            X_subset = X[:, subset]
            score = criteria_J(X_subset, y)
            
            results.append({
                "features": subset,
                "score": score
            })
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]    

results = feature_search(X_white, y_white, top_k=50)

for i, res in enumerate(results):
    print(f"{i+1}: Features={res['features']}, J={res['score']:.4f}")

best_sets = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4, 5, 7, 8, 9, 10),
    (0, 1, 2, 3, 5, 7, 8, 9, 10),
    (0, 1, 3, 4, 5, 7, 8, 9, 10),
    (0, 1, 3, 5, 7, 8, 9, 10),
]
#for features in best_sets:
 #   k_nn = 0.5 
  #  
 #   X_sub = X_white[:, features]
 #   model_class = lambda: ParzenGaussian(sigma=k_nn)
 #   mean, std, avg_time, tot_time = repeated_holdout(LDA, X_sub, y_white, n_runs=20, test_size=0.2, seed_offset=0)
 #  print_acc_report(mean, std, avg_time, tot_time, title=f"{"White Wine"} - {"KNN"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")            

best_set = (0, 1, 3, 4, 5, 6, 7, 8, 9, 10)
X_sub = X_white[:, best_set]

#model_class_1 = lambda: ParzenGaussian(sigma=0.5)
#model_class_2 = lambda: K_Nearest_Neighbor(k=1)

#ean, std, avg_time, tot_time = repeated_holdout(QDA, X_sub, y_white, n_runs=20, test_size=0.2, seed_offset=0)
#rint_acc_report(mean, std, avg_time, tot_time, title=f"{"White Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  

#mean, std, avg_time, tot_time = repeated_kfold(QDA, X_sub, y_white, k=5, n_runs=20, seed_offset=0)
#print_acc_report(mean, std, avg_time, tot_time, title=f"{"White Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  
 
#mean, std, avg_time, tot_time = repeated_kfold(QDA, X_sub, y_white, k=10, n_runs=20, seed_offset=0)
#print_acc_report(mean, std, avg_time, tot_time, title=f"{"White Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  
 
#mean, std, avg_time, tot_time = loocv(QDA, X_sub, y_white)
#print_acc_report(mean, std, avg_time, tot_time, title=f"{"White Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  

# New Feature
# x_new = (residual alcohol * density) / alcohol
f1 = ((X_red[:, 3] * X_red[:, 7]) / X_red[:, 10])

alcohol = X_red[:, 10]
density = (1 - X_red[:, 7])
f2 = ((alcohol * density)**2)
X_new = np.column_stack((X_red, f1, f2))

scaler = StandardScaler()

X_new_scaled = scaler.fit_transform(X_new)



model_class_1 = lambda: ParzenGaussian(sigma=0.5)
model_class_2 = lambda: K_Nearest_Neighbor(k=1)
 
mean, std, avg_time, tot_time = loocv(model_class_1, X_new_scaled, y_red)
print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  
y_true, y_pred = loocv_cm(model_class_1, X_new_scaled, y_red)
cm, classes = confusion_matrix(y_true, y_pred)
print_cm_report(cm, classes, "CM - PW - LOOCV \nConfusion Matrix")


