import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

def build_and_predict(train_path, test_path, out_path, degree):
    print(f'\nTraining model for {train_path}')
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    X_train = df_train.drop('y', axis=1).values
    y_train = df_train['y'].values
    X_test  = df_test.values

    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = np.logspace(-3, 3, 30)   # 30 alphas is enough

    #Ridge
    print('  Evaluating Ridge')
    pipe_ridge = Pipeline([
        ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  RidgeCV(alphas=alphas, cv=5))
    ])
    cv_mse_ridge = -cross_val_score(
        pipe_ridge, X_train, y_train, cv=kf,
        scoring='neg_mean_squared_error', n_jobs=-1
    ).mean()
    print(f'  Ridge  CV MSE: {cv_mse_ridge:.4f}')

    #LASSO
    print('  Evaluating LASSO')
    pipe_lasso = Pipeline([
        ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model',  LassoCV(alphas=alphas, cv=5, max_iter=5000, n_jobs=-1))
    ])
    cv_mse_lasso = -cross_val_score(
        pipe_lasso, X_train, y_train, cv=kf,
        scoring='neg_mean_squared_error', n_jobs=-1
    ).mean()
    print(f'  LASSO  CV MSE: {cv_mse_lasso:.4f}')

    #we pick best of the two
    if cv_mse_ridge <= cv_mse_lasso:
        chosen_name = 'Ridge'
        best_pipe   = pipe_ridge
        best_cv_mse = cv_mse_ridge
    else:
        chosen_name = 'LASSO'
        best_pipe   = pipe_lasso
        best_cv_mse = cv_mse_lasso

    print(f'  Selected: {chosen_name}')

    # Train final model on all training data
    best_pipe.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, best_pipe.predict(X_train))

    y_pred = best_pipe.predict(X_test)
    df_test['y'] = y_pred
    df_test.to_csv(out_path, index=False)

    print(f'  Degree:         {degree}')
    print(f'  Training MSE:   {train_mse:.4f}')
    print(f'  CV MSE:         {best_cv_mse:.4f}')
    lam = getattr(best_pipe.named_steps['model'], 'alpha_', None)
    if lam is not None:
        print(f'  Optimal lambda: {lam:.6f}')
    print(f'  Saved predictions to {out_path}.')

train_var1_path = '/home/adity/BT2024035/BT2024035_train_var1.csv'
test_var1_path  = '/home/adity/BT2024035/BT2024035_test_var1.csv'
pred_var1_path  = '/home/adity/BT2024035/BT2024035_pred_var1.csv'

train_var2_path = '/home/adity/BT2024035/BT2024035_train_var2.csv'
test_var2_path  = '/home/adity/BT2024035/BT2024035_test_var2.csv'
pred_var2_path  = '/home/adity/BT2024035/BT2024035_pred_var2.csv'

build_and_predict(train_var1_path, test_var1_path, pred_var1_path, degree=5)
build_and_predict(train_var2_path, test_var2_path, pred_var2_path, degree=11)
