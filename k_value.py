############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# k_value.py
# Used to find best k_value for KNN
# for each dataset
############################################################

from cv_accuracy import repeated_holdout, repeated_kfold, loocv, holdout_cv, kfold_cv, print_acc_report
from classifiers import K_Nearest_Neighbor, ParzenGaussian
from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red
import numpy as np
import matplotlib.pyplot as plt


k_values = range(1, 50)
best_k = None
best_score = -np.inf

results = {}

for k in k_values:   
    model_class = lambda: K_Nearest_Neighbor(k=k)
    mean, std, avg_time, tot_time = repeated_holdout(model_class, X_red, y_red, n_runs=20, test_size=0.2, seed_offset=0)

    score = mean - std
    
    results[k] = {
        "mean": mean,
        "std": std,
        "score": score
    }
    
    if score > best_score:
        best_score = score
        best_k = k

print("Best k:", best_k)
print("Best score:", best_score) 

ks = list(results.keys())
scores = [results[k]["score"] for k in ks]

plt.plot(ks, scores)
plt.xlabel("k")
plt.ylabel("Score")
plt.title("KNN Score vs k")
plt.show()
plt.close()

    



