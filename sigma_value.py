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


sigma_values = np.arange(0.1, 5.1, 0.1)
best_sigma = None
best_score = -np.inf

results = {}

for sigma in sigma_values:   
    sigma = round(sigma, 2)
    print(f"sigma={sigma}")
    
    model_class = lambda sigma=sigma: ParzenGaussian(sigma=sigma)
    mean, std, avg_time, tot_time = repeated_holdout(model_class, X_white, y_white, n_runs=20, test_size=0.2, seed_offset=0)

    score = mean - std
    
    results[sigma] = {
        "mean": mean,
        "std": std,
        "score": score
    }
    
    if score > best_score:
        best_score = score
        best_sigma = sigma

print("Best sigma:", best_sigma)
print("Best score:", best_score) 

ks = list(results.keys())
scores = [results[k]["score"] for k in ks]

plt.plot(ks, scores)
plt.xlabel("sigma")
plt.ylabel("Score")
plt.title("Parzen Windows Score vs sigma")
plt.show()
plt.close()

    



