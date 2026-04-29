############################################################
# ECE4720 Final Project
# Machine Learning and Pattern Recognition
# Thomas Joswiak
# main.py
# Main workflow
# Mostly function calls and printing
############################################################

import pandas as pd

# Load datasets
raisin_ds = pd.read_excel("Raisin_Dataset.xlsx")
red_wine_ds = pd.read_csv("winequality-red.csv", sep=';')
white_wine_ds = pd.read_csv("winequality-white.csv", sep=';')

X_raisin = raisin_ds.iloc[:, 0:7]
y_raisin = raisin_ds.iloc[:, 7]

X_white = white_wine_ds.iloc[:, :-1]
y_white = white_wine_ds.iloc[:, -1]

X_red = red_wine_ds.iloc[:, :-1]
y_red = red_wine_ds.iloc[:, -1]

# Convert to numpy matricies
X_raisin = X_raisin.to_numpy()
y_raisin = y_raisin.to_numpy()
raisin_samples, raisin_features = X_raisin.shape

X_white = X_white.to_numpy()
y_white = y_white.to_numpy()
white_samples, white_features = X_white.shape

X_red = X_red.to_numpy()
y_red = y_red.to_numpy()
red_samples, red_features = X_red.shape
