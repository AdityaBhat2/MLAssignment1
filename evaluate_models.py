import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

train_var1_path = "/home/adity/BT2024035/BT2024035_train_var1.csv"
train_var2_path = "/home/adity/BT2024035/BT2024035_train_var2.csv"


def evaluate_poly(X, y, max_degree):
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 3, 30)

    results = []

    for deg in range(1, max_degree + 1):
        pipe_ridge = Pipeline([
            ('poly',   PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model',  RidgeCV(alphas=alphas, cv=5))
        ])
        pipe_lasso = Pipeline([
            ('poly',   PolynomialFeatures(degree=deg, include_bias=False)),
            ('scaler', StandardScaler()),
            ('model',  LassoCV(alphas=alphas, cv=5, max_iter=5000, n_jobs=-1))
        ])

        mse_ridge = -cross_val_score(pipe_ridge, X, y, cv=kf,
                                     scoring='neg_mean_squared_error', n_jobs=-1).mean()
        mse_lasso = -cross_val_score(pipe_lasso, X, y, cv=kf,
                                     scoring='neg_mean_squared_error', n_jobs=-1).mean()

        winner  = "Ridge" if mse_ridge <= mse_lasso else "LASSO"
        best_cv = min(mse_ridge, mse_lasso)
        results.append((deg, mse_ridge, mse_lasso, winner, best_cv))

    # Find the best degree
    best_idx = min(range(len(results)), key=lambda i: results[i][4])

    # Print table
    print(f"{'Degree':>6} | {'Ridge CV MSE':>12} | {'LASSO CV MSE':>12} | Winner")
    print("-" * 54)
    for i, (deg, mse_r, mse_l, winner, best_cv) in enumerate(results):
        marker = " <-- best" if i == best_idx else ""
        print(f"{deg:>6} | {mse_r:>12.4f} | {mse_l:>12.4f} | {winner}{marker}")

    best = results[best_idx]
    print(f"\nBest => Degree {best[0]}, {best[3]}, CV MSE = {best[4]:.4f}")
    return best[0], best[3]


print("VAR 1")
df1 = pd.read_csv(train_var1_path)
X1, y1 = df1.drop('y', axis=1).values, df1['y'].values
evaluate_poly(X1, y1, max_degree=10)

print("\nVAR 2")
df2 = pd.read_csv(train_var2_path)
X2, y2 = df2.drop('y', axis=1).values, df2['y'].values
evaluate_poly(X2, y2, max_degree=20)
