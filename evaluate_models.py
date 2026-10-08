import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

train_var1_path = "/home/adity/BT2024035/BT2024035_train_var1.csv"
train_var2_path = "/home/adity/BT2024035/BT2024035_train_var2.csv"

def evaluate_poly(X, y, max_degree, use_ridge=True):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_deg = 1
    best_score = -np.inf
    
    for deg in range(1, max_degree + 1):
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        model = Ridge(alpha=1.0) if use_ridge else LinearRegression()
        
        scores = cross_val_score(model, X_poly, y, cv=kf, scoring='neg_mean_squared_error')
        mse = -scores.mean()
        
        r2_scores = cross_val_score(model, X_poly, y, cv=kf, scoring='r2')
        r2 = r2_scores.mean()
        
        print(f"Degree {deg}: MSE = {mse:.4f}, R2 = {r2:.4f}")
        if r2 > best_score:
            best_score = r2
            best_deg = deg
            
    return best_deg

print("=== VAR 1 ===")
df1 = pd.read_csv(train_var1_path)
X1 = df1.drop('y', axis=1).values
y1 = df1['y'].values
evaluate_poly(X1, y1, 8, use_ridge=True)

print("\n=== VAR 2 ===")
df2 = pd.read_csv(train_var2_path)
X2 = df2.drop('y', axis=1).values
y2 = df2['y'].values
evaluate_poly(X2, y2, 10, use_ridge=True)
