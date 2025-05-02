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
from sklearn.neural_network import MLPRegressor
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

if __name__ == '__main__':
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    i = 1
    data_path, file_name = "data", f"magpie_{target[i]}.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    df.drop_duplicates(keep="first", inplace=True)
    df.dropna(inplace=True)
    # 这里不根据四分位数去除极大极小值
    df2 = df.drop(["formula", "composition_obj", "Weight Fraction", "Atomic Fraction"], axis=1)
    print(df2.describe(), df.shape)
    features = [i for i in df2.columns if i not in target]
    X = df2[features]
    Y = df2[target[i]]

    # 数据划分与标准化
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    X_test_std = std.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)
    print(X_train_std, X_test_std)

    # 粗糙选择最佳特征（可跳过）
    k = len(features)
    selector = SelectKBest(f_regression, k=k)
    best = selector.fit_transform(X_train, Y_train)
    mask = selector.get_support()  # 布尔数组，表示哪些被选中
    selected_features = [name for name, keep in zip(features, mask) if keep]
    # print("被选中的特征:", selected_features)
    X_train_std = X_train_std[selected_features]
    X_test_std = X_test_std[selected_features]

    # 初步建模
    model_dict = {
        # "lgb": lgb.LGBMRegressor(),
        "svr": SVR(C=100, gamma=0.1, kernel="rbf"),
        # "ridge": Ridge(random_state=1),
        # "lasso": Lasso(random_state=1),
        # "elasticnet": ElasticNet(random_state=1),
        # "linear": LinearRegression(),
        # "xgb": XGBRegressor(random_state=1),
        # "bayesian": BayesianRidge(),
        # "randomforest": RandomForestRegressor(random_state=1),
        # "adaboost": AdaBoostRegressor(random_state=1),
        # "extratrees": ExtraTreesRegressor(random_state=1),
        # "gradientboost": GradientBoostingRegressor(random_state=1),
        # "ann": MLPRegressor(random_state=1)
        }
    best_score = -100
    best_model = ""
    for model_name, model in model_dict.items():
        model.fit(X_train_std, Y_train)
        Y_pred = model.predict(X_test_std)
        cv_predict = cross_val_predict(model, X, Y, cv=10)
        score = r2_score(Y, cv_predict)
        if score > best_score:
            best_score = score
            best_model = model_name
        print(model_name, r2_score(Y_test, Y_pred), r2_score(Y, cv_predict))
        plt.scatter(Y_test, Y_pred, color="blue", s=5)
        plt.scatter(Y, cv_predict, color="red", s=5)
        for i in range(len(Y)):
            plt.text(list(Y)[i], list(cv_predict)[i], str(i), fontsize=8, ha='center', va='bottom', color='black')
        plt.plot([min(Y), max(Y)], [min(Y), max(Y)],
                 color='red', linestyle='--', label='y = x')
        plt.xlabel("True Value")
        plt.ylabel("Predict Value")
        plt.show()
    model_dict[best_model].fit(X_train_std, Y_train)
    cv_predict = cross_val_predict(model_dict[best_model], X, Y, cv=10)
    cv_score = CVS(model_dict[best_model], X, Y, cv=10)
    print(cv_score)
    print(best_model, best_score, pearsonr(Y, cv_predict))
    # print(Y.to_string())