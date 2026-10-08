import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

def build_and_predict(train_path, test_path, out_path, degree):
    print(f"\n--- Training model for {train_path} ---")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.drop('y', axis=1).values
    y_train = df_train['y'].values
    X_test = df_test.values

    # Using StandardScaler is critical for Ridge to penalize features equally
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    # RidgeCV tests multiple alphas to find the optimal regularization strength
    model = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
        
    pipe = Pipeline([
        ('poly', poly),
        ('scaler', scaler),
        ('model', model)
    ])
    
    # Calculate CV MSE to estimate test performance
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    
    # Train final model on all data
    pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, pipe.predict(X_train))
    
    y_pred = pipe.predict(X_test)
    
    df_test['y'] = y_pred
    df_test.to_csv(out_path, index=False)
    print(f"Optimal Degree Used: {degree}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Estimated Test (CV) MSE: {cv_mse:.4f}")
    if hasattr(pipe.named_steps['model'], 'alpha_'):
        print(f"Optimal Alpha Selected: {pipe.named_steps['model'].alpha_:.4f}")
    print(f"Saved predictions to {out_path}.")

train_var1_path = "/home/adity/BT2024035/BT2024035_train_var1.csv"
test_var1_path = "/home/adity/BT2024035/BT2024035_test_var1.csv"
pred_var1_path = "/home/adity/BT2024035/BT2024035_pred_var1.csv"

train_var2_path = "/home/adity/BT2024035/BT2024035_train_var2.csv"
test_var2_path = "/home/adity/BT2024035/BT2024035_test_var2.csv"
pred_var2_path = "/home/adity/BT2024035/BT2024035_pred_var2.csv"

# Degree 5 is optimal for var1
build_and_predict(train_var1_path, test_var1_path, pred_var1_path, degree=5)

# Degree 10 is optimal for var2
build_and_predict(train_var2_path, test_var2_path, pred_var2_path, degree=10)
