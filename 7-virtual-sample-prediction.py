import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pickle

model_dict = {
    "lgb": lgb.LGBMRegressor(random_state=1),
    "svr": SVR(C=100, gamma=0.1, kernel="rbf"),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elasticnet": ElasticNet(),
    "linear": LinearRegression(),
    "xgb": XGBRegressor(random_state=1),
    "bayesian": BayesianRidge(),  # 测试集效果奇差
    "randomforest": RandomForestRegressor(random_state=1),
    "adaboost": AdaBoostRegressor(random_state=1, n_estimators=100, learning_rate=0.1),
    "extratrees": ExtraTreesRegressor(random_state=1),
    "gradientboost": GradientBoostingRegressor(random_state=1)
}

if __name__ == '__main__':
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    i = 0
    model_name = [j for j in model_dict.keys()]
    print(model_name)
    with open(f'model_{target[i]}_{model_name[10]}.pkl', 'rb') as f:
        model = pickle.load(f)