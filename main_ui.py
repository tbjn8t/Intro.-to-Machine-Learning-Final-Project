############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# main_ui.py
# UI used to mavigate datasets, accuracy/CM,
# parameters, classifiers, visualizers, etc
############################################################

from data import X_raisin, y_raisin, X_white, y_white, X_red, y_red
from data import raisin_samples, raisin_features
from data import white_samples, white_features
from data import red_samples, red_features

from cv_accuracy import repeated_holdout, repeated_kfold, loocv, holdout_cv, kfold_cv, print_acc_report
from classifiers import NaiveBayes, K_Nearest_Neighbor, ParzenGaussian, LDA, QDA
from cv_confusion import holdout_cm, kfold_cm, loocv_cm
from cv_confusion import confusion_matrix, print_cm_report
from PCA import pca_2d, tsne_2d, umap_2d

def run_ui():
    
    while True: 
        # UI header
        print("\n" + "="*40)
        print("ML FINAL PROJECT")
        print("THOMAS JOSWIAK")
        print("="*40)
    
        # Dataset selection
        datasets = {
            "1": ("Raisin", X_raisin, y_raisin),
            "2": ("White WIne", X_white, y_white),
            "3": ("Red Wine", X_red, y_red)
        }
        print("\nSelect dataset:")
        print(f"1 - Raisin - {raisin_samples} samples and {raisin_features} features")
        print(f"2 - White Wine - {white_samples} samples and {white_features} features")
        print(f"3 - Red Wine - {red_samples} samples and {red_features} features")
        ds_in = input("Choice: ")
        name, X, y = datasets[ds_in]
        
        # Model selection
        models = {
            "1": ("Naive Bayes", NaiveBayes),
            "2": ("KNN", K_Nearest_Neighbor),
            "3": ("Parzen", ParzenGaussian),
            "4": ("LDA", LDA),
            "5": ("QDA", QDA),
        }
        print("\nSelect classifier:")
        print("1 - Naive Bayes")
        print("2 - KNN")
        print("3 - Parzen Window (Gaussian)")
        print("4 - LDA")
        print("5 - QDA")
        print("6 - Plot Samples")
        model_in = input("Choice: ")
        
        # Plotting sub selection tree
        if model_in == "6":
            print("\nSelect Plot Methjod:")
            print("1 - PCA")
            print("2 - t-SNE")
            print("3 - UMAP")
            
            plot_in = input("Choice: ")
            if plot_in == "1":
                pca_2d(X, y, title=f"PCA 2D - {name}")
            elif plot_in == "2":
                perplexity = int(input("Perplexity value {ex. 30}: "))
                tsne_2d(X, y, title=f"{name} - t-SNE 2D", perplexity=perplexity)
            elif plot_in == "3":
                n_neighbors = int(input("N_Neighbors value {ex. 15}: "))
                umap_2d(X, y, title=f"{name} - UMAP", n_neighbors=n_neighbors)
            continue
        
        model_name, model_class = models[model_in]
        
        # Request k value for KNN when chosen
        if model_in == "2":
            k_nn = int(input("Enter k value for KNN (ex. 5, 10): "))
            model_class = lambda: K_Nearest_Neighbor(k=k_nn)
        elif model_in == "3":
            sigma_pw = float(input("Enter σ value for Parzen Window (ex. 0.1, 1): "))
            model_class = lambda: ParzenGaussian(sigma=sigma_pw)
        
        # CV method selection
        cv_methods = {
            "1": "Holdout",
            "2": "K-Fold",
            "3": "LOOCV",
        }
        print("\nSelect CV method:")
        print("1 - Holdout")
        print("2 - K-Fold")
        print("3 - LOOCV")
        cv_in = input("Choice: ")
        
        cv_name = cv_methods[cv_in]
        
        # Output selection
        print("\nSelect output:")
        print("1 - Accuracy")
        print("2 - Confusion Matrix")
        out_in = input("Choice: ")
        
        # Run selected sequence
        
        # If looking for accuracy (multiple runs used)
        if out_in == "1":
            if cv_in == "1":
                # Holdout acc
                n_runs = int(input("# of runs {ex. 20}: "))
                test_size = float(input("Dataset parition {ex. 0.2=20%}: "))
                seed_offset = int(input("Seed offset (starting seed -> num_runs) {ex. 0}: "))
                mean, std, avg_time, tot_time = repeated_holdout(model_class, X, y, n_runs=n_runs, test_size=test_size, seed_offset=seed_offset)
                print_acc_report(mean, std, avg_time, tot_time, title=f"{name} - {model_name} - {cv_name}\n{n_runs} runs - {test_size} size\nAccuracy Results")            
            elif cv_in == "2":
                # Kfold acc
                n_runs = int(input("# of runs {ex. 20}: "))
                k = int(input("Number of folds {ex. 5}: "))
                seed_offset = int(input("Seed offset (starting seed -> num_runs) {ex. 0}: "))
                mean, std, avg_time, tot_time = repeated_kfold(model_class, X, y, k=k, n_runs=n_runs, seed_offset=seed_offset)
                print_acc_report(mean, std, avg_time, tot_time, title=f"{name} - {model_name} - {cv_name}\n{n_runs} runs - k={k}\nAccuracy Results")            
            elif cv_in == "3":
                # LOOCV acc
                mean, std, avg_time, tot_time = loocv(model_class, X, y)
                print_acc_report(mean, std, avg_time, tot_time, title=f"{name} - {model_name} - {cv_name}\nAccuracy Results")            
        
        # If looking for confusion matrix (one run used)
        elif out_in == "2":
            if cv_in == "1":
                # Holdout CM
                test_size = float(input("Test size {ex. 0.2}: "))
                random_state = int(input("Random seed {ex. 1, 42}: "))
                y_true, y_pred = holdout_cm(model_class, X, y, test_size=test_size, shuffle=True, random_state=random_state)
                cm, classes = confusion_matrix(y_true, y_pred)
                print_cm_report(cm, classes, title=f"{name} - {model_name} - {cv_name} \nConfusion Matrix")            
            elif cv_in == "2":
                # KFold CM
                k_fold = int(input("Number of folds {ex. 5}: "))
                random_state = int(input("Random seed {ex. 1, 42}: "))
                y_true, y_pred = kfold_cm(model_class, X, y, k=k_fold, shuffle=True, random_state=random_state)
                cm, classes = confusion_matrix(y_true, y_pred)
                print_cm_report(cm, classes, title=f"{name} - {model_name} - {cv_name} \nConfusion Matrix")            
            elif cv_in == "3":
                # LOOCV CM
                y_true, y_pred = loocv_cm(model_class, X, y)
                cm, classes = confusion_matrix(y_true, y_pred)
                print_cm_report(cm, classes, title=f"{name} - {model_name} - {cv_name} \nConfusion Matrix")
        
        # Run again
        print("-" * 40)
        again = input("Run another experiment? (y/n): ").strip().lower()
        
        if again != "y":
            print("Exiting program.")
            break
    

run_ui()