import joblib
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
from sklearn.linear_model import Lasso, LassoLars
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
from util.alloys_features import formula_to_ratio_dataset

if __name__ == '__main__':
    melting_point_best = ['0-norm', '2-norm', 'MagpieData minimum Number', 'MagpieData mean Number',
                        'MagpieData minimum MendeleevNumber', 'MagpieData avg_dev MendeleevNumber', 'MagpieData minimum CovalentRadius',
                        'MagpieData range Electronegativity', 'MagpieData minimum NsValence', 'MagpieData mean NsValence',
                        'MagpieData maximum SpaceGroupNumber', 'compound possible', 'Yang delta', 'Yang omega', 'APE mean', 'Mixing enthalpy',
                        'Mean cohesive energy']
    enthalpy_best = ['MagpieData minimum MendeleevNumber', 'MagpieData maximum Electronegativity', 'MagpieData range Electronegativity',
                     'MagpieData minimum NValence', 'MagpieData minimum GSvolume_pa', 'MagpieData range SpaceGroupNumber',
                     'MagpieData mode SpaceGroupNumber', 'compound possible']
    best_all = [i for i in list(set(melting_point_best + enthalpy_best)) if i not in ['MagpieData mean Number', 'MagpieData minimum CovalentRadius', 'MagpieData minimum NsValence', 'MagpieData mean NsValence']]
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    i = 1
    data_path, file_name = "data", f"magpie_{target[i]}_feature_selected-1.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    df.drop_duplicates(keep="first", inplace=True)
    df.dropna(inplace=True)

    # 添加原子比特征
    ratio = pd.read_csv(f"data/{target[i]}_formula.csv")
    ratio_all, element_columns = formula_to_ratio_dataset(ratio)
    ratio_all = ratio_all.iloc[:, 1:]
    print(element_columns)


    features = [str(i) for i in df.columns if i not in target]

    best_feature = [melting_point_best, enthalpy_best, best_all, features]
    X = pd.concat([df[best_feature[i]], ratio_all], axis=1)
    # X.columns = [str(col) for col in X.columns]
    Y = df[target[i]]

    # 数据划分与标准化
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=153)  # 15
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    joblib.dump(std, f"standard_scaler_{target[i]}.pkl")  # 保存归一化器
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    X_test_std = std.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)
    # print(X_train_std, X_test_std)

    # 筛选后特征建模
    model_dict = {
        # "lgb": lgb.LGBMRegressor(random_state=1),
        # "svr": SVR(C=100, gamma=0.1, kernel="rbf"),
        # "ridge": Ridge(),
        # "lasso": Lasso(),
        # "elasticnet": ElasticNet(),
        # "linear": LinearRegression(),
        # "xgb": XGBRegressor(random_state=1),
        # # "bayesian": BayesianRidge(),  # 测试集效果奇差
        # "randomforest": RandomForestRegressor(random_state=1),
        # "adaboost": AdaBoostRegressor(random_state=1, n_estimators=100, learning_rate=0.1),
        "extratrees": ExtraTreesRegressor(random_state=1),
        # "gradientboost": GradientBoostingRegressor(random_state=1),
        # "lassolars": LassoLars()
    }
    best_score = 0
    best_model = ""
    for model_name, model in model_dict.items():
        model.fit(X_train_std, Y_train)
        Y_pred = model.predict(X_test_std)
        Y_train_pred = model.predict(X_train_std)
        cv_predict = cross_val_predict(model, X_train_std, Y_train, cv=10)
        score = r2_score(Y_train, cv_predict)
        if score > best_score:
            best_score = score
            best_model = model_name
        print(model_name, r2_score(Y_test, Y_pred), r2_score(Y_train, cv_predict))
        # print(model_name, r2_score(Y_test, Y_pred), r2_score(Y, cv_predict))
        # plt.scatter(Y_test, Y_pred, color="blue", s=5)
        # plt.scatter(Y, cv_predict, color="red", s=5)
        # for i in range(len(Y)):
        #     plt.text(list(Y)[i], list(cv_predict)[i], str(i), fontsize=8, ha='center', va='bottom', color='black')
        # plt.plot([min(Y), max(Y)], [min(Y), max(Y)],
        #          color='red', linestyle='--', label='y = x')
        # plt.xlabel("True Value")
        # plt.ylabel("Predict Value")
        # plt.show()
    model = model_dict[best_model].fit(X_train_std, Y_train)
    with open(f'models/model_{target[i]}_{best_model}.pkl', 'wb') as f:
        pickle.dump(model, f)

    Y_test_pred = model.predict(X_test_std)
    Y_train_pred = model.predict(X_train_std)
    plt.scatter(Y_train, Y_train_pred, color="red", s=5)
    plt.scatter(Y_test, Y_test_pred, color="blue", s=5)
    x_range = np.linspace(plt.xlim()[0], plt.xlim()[1])
    y_range = np.linspace(plt.ylim()[0], plt.ylim()[1])
    len_x = (plt.xlim()[1] - plt.xlim()[0])
    plt.plot(x_range, x_range, "r--", linewidth=0.25)
    plt.xlabel("True Value")
    plt.ylabel("Predict Value")
    plt.text(0.5 * len_x, 0.8 * plt.xlim()[1], f"R^2: {r2_score(Y_test, Y_test_pred):.4f}", fontsize=12, ha='center', va='bottom', color='black')
    plt.show()


    cv_predict = cross_val_predict(model_dict[best_model], X_train_std, Y_train, cv=10)
    plt.clf()
    plt.scatter(Y_train, cv_predict, color="red", s=5)
    plt.plot(x_range, x_range, "r--", linewidth=0.25)
    plt.xlabel("True Value")
    plt.ylabel("Predict Value")
    plt.text(0.5 * len_x, 0.8 * plt.xlim()[1], f"R^2: {r2_score(Y_train, cv_predict):.4f}", fontsize=12, ha='center', va='bottom', color='black')
    plt.show()
    print(best_model, best_score, r2_score(Y_train, Y_train_pred), pearsonr(Y_train, cv_predict))
    # print(Y.to_string())