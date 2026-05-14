############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# feature_selection.py
# Uses J1 criteria to compute class separable feature combinations
# Tests the best J1 criteria using each classifier/CV
# Then, created two new features and adds them to the datasets
# Tests these new datasets for accuracy and CM matrix
############################################################

from cv_accuracy import repeated_holdout, repeated_kfold, loocv, holdout_cv, kfold_cv, print_acc_report
from classifiers import NaiveBayes, K_Nearest_Neighbor, ParzenGaussian, LDA, QDA
from cv_confusion import holdout_cm, kfold_cm, loocv_cm
from cv_confusion import confusion_matrix, print_cm_report
from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools

# J1 Criteria computation
# J1 = tr(Sw^-1 * Sb)
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

# Iterates through all possible feature combinations
# Calculates J1 Criteria using Sw and Sb for all combinations
# Stored and ranked
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

results_white = feature_search(X_white, y_white, top_k=50)
results_red = feature_search(X_red, y_red, top_k=50)

# Print all J1 Criteria computed
for i, res in enumerate(results_red):
    print(f"{i+1}: Features={res['features']}, J={res['score']:.4f}")

# Top 8 White Wine J1 Criteria
best_sets_white = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4, 5, 7, 8, 9, 10),
    (0, 1, 2, 3, 5, 7, 8, 9, 10),
    (0, 1, 3, 4, 5, 7, 8, 9, 10),
    (0, 1, 3, 5, 7, 8, 9, 10),
]

# Top 8 Red Wine J1 Criteria
best_sets_red = [
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4, 5, 6, 7, 9, 10),
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 3, 4, 6, 7, 8, 9, 10),
    (0, 1, 2, 4, 5, 6, 7, 8, 9, 10),
    (0, 1, 2, 3, 4, 5, 6, 8, 9, 10),
]

# Test top 8 J1 criteria using Kfold=10
# Test across all classifiers, White and Red Wine datasets
#for features in best_sets_red:
    #sigma = 0.7 
    #k=1

   # X_sub_white = X_white[:, features]
   # X_sub_red = X_red[:, features]
    
   # model_class_1 = lambda: K_Nearest_Neighbor(k=k)
   # model_class_2 = lambda: ParzenGaussian(sigma=sigma)
  #  mean, std, avg_time, tot_time = repeated_kfold(model_class_2, X_sub_red, y_red, k=10, n_runs=20, seed_offset=0)
  #  print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"KF=10"}\nAccuracy Results")            

# Best J1 Criteria for each dataset
# Determined by test loop above
best_set_white = (0, 1, 3, 5, 7, 8, 9, 10)
best_set_red = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
X_sub_white = X_white[:, best_set_white]
X_sub_red = X_red[:, best_set_red]

model_class_1 = lambda: ParzenGaussian(sigma=0.7)
model_class_2 = lambda: K_Nearest_Neighbor(k=1)

# Tests the J1 Criteria for each classifier, for each CV method
mean, std, avg_time, tot_time = repeated_holdout(model_class_1, X_sub_red, y_red, n_runs=20, test_size=0.2, seed_offset=0)
print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"Holdout"}\n{20} runs - {0.2} size\nAccuracy Results")  

mean, std, avg_time, tot_time = repeated_kfold(model_class_1, X_sub_red, y_red, k=5, n_runs=20, seed_offset=0)
print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"K=5"}\n{20} runs - {0.2} size\nAccuracy Results")  
 
mean, std, avg_time, tot_time = repeated_kfold(model_class_1, X_sub_red, y_red, k=10, n_runs=20, seed_offset=0)
print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"K=10"}\n{20} runs - {0.2} size\nAccuracy Results")  
 
mean, std, avg_time, tot_time = loocv(model_class_1, X_sub_red, y_red)
print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"NB"} - {"LOO"}\n{20} runs - {0.2} size\nAccuracy Results")  

# New Feature
# x_new = (residual alcohol * density) / alcohol
f1 = ((X_white[:, 3] * X_white[:, 7]) / X_white[:, 10])

alcohol = X_white[:, 10]
density = (1 - X_white[:, 7])

# x_new = alcohol(1 - density)
# f2 = x_new^2
f2 = ((alcohol * density)**2)
X_new = np.column_stack((X_white, f1, f2))

scaler = StandardScaler()

X_new_scaled = scaler.fit_transform(X_new)

# Test new features for each classifier using LOOCV
# Print out accuracy table and CM table
model_class_1 = lambda: ParzenGaussian(sigma=0.5)
model_class_2 = lambda: K_Nearest_Neighbor(k=1)
 
mean, std, avg_time, tot_time = loocv(model_class_1, X_new_scaled, y_white)
#print_acc_report(mean, std, avg_time, tot_time, title=f"{"Red Wine"} - {"KNN"} - {"LOO"}\nAccuracy Results")  
y_true, y_pred = loocv_cm(model_class_1, X_new_scaled, y_white)
cm, classes = confusion_matrix(y_true, y_pred)
#print_cm_report(cm, classes, "CM - PW - LOOCV \nConfusion Matrix")


