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
    "gradientboost": GradientBoostingRegressor(random_state=1),
    "lassolars": LassoLars()
}

melting_point_best = ['0-norm', '2-norm', 'MagpieData minimum Number', 'MagpieData mean Number',
                      'MagpieData minimum MendeleevNumber', 'MagpieData avg_dev MendeleevNumber',
                      'MagpieData minimum CovalentRadius',
                      'MagpieData range Electronegativity', 'MagpieData minimum NsValence', 'MagpieData mean NsValence',
                      'MagpieData maximum SpaceGroupNumber', 'compound possible', 'Yang delta', 'Yang omega',
                      'APE mean', 'Mixing enthalpy',
                      'Mean cohesive energy']
enthalpy_best = ['MagpieData minimum MendeleevNumber', 'MagpieData maximum Electronegativity',
                 'MagpieData range Electronegativity',
                 'MagpieData minimum NValence', 'MagpieData minimum GSvolume_pa', 'MagpieData range SpaceGroupNumber',
                 'MagpieData mode SpaceGroupNumber', 'compound possible']

if __name__ == '__main__':
    # target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    # i = 0
    # model_name = [j for j in model_dict.keys()]
    # print(model_name)
    # with open(f'models/model_{target[i]}_{model_name[10]}.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # # features = pd.read_csv("data/magpie_virtual_sample.csv")  # 高通量的
    # num_samples = 50000
    # features = pd.read_csv(f"data/magpie_virtual_sample_WAE_{num_samples}.csv")  # WAE的
    # best_feature = [melting_point_best, enthalpy_best, features, features]
    #
    # std = joblib.load(f"standard_scaler_{target[i]}.pkl")
    #
    # X_std = std.transform(features[best_feature[i]])
    # Y_predict = model.predict(X_std)
    # result = pd.DataFrame({"formula": features["formula"], "predict": Y_predict})
    # result.to_csv(f"data/virtual_samples_result_{target[i]}_{model_name[10]}_WAE_{num_samples}.csv", index=False)

    # 进一步筛选
    # features = pd.read_csv(f"data/magpie_wae50000_filter.csv")
    # best_feature = [melting_point_best, enthalpy_best, features, features]
    # target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    # i = 1
    # model_name = [j for j in model_dict.keys()]
    # print(model_name)
    # with open(f'models/model_{target[i]}_{model_name[10]}.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # std = joblib.load(f"standard_scaler_{target[i]}.pkl")
    #
    # X_std = std.transform(features[best_feature[i]])
    # Y_predict = model.predict(X_std)
    # result = pd.DataFrame({"formula": features["formula"], "predict": Y_predict})
    # result.to_csv(f"data/wae50000_filter_{target[i]}_{model_name[10]}.csv", index=False)

    # 添加原子比特征
    ratio = pd.read_csv(f"data/virtual_samples_lmp_alloys_formula_Ga_In_Sn.csv")
    ratio_all, element_columns = formula_to_ratio_dataset(ratio)
    ratio_all = ratio_all.iloc[:, 1:]
    new_order = ['Sn', 'In', 'Ga']  # 你想要的新列顺序
    ratio_all = ratio_all[new_order]
    # 与模型特征一致
    new_columns = ['Zn', 'Cd', 'Al', 'Ag', 'Pb', 'Bi', 'Cu', 'Ti']
    ratio_all[new_columns] = 0
    print(ratio_all)

    # 对于特定元素样本
    features = pd.read_csv(f"data/magpie_virtual_sample_Ga_In_Sn.csv")
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    best_feature = [melting_point_best, enthalpy_best, features, features]
    i = 0
    model_name = [j for j in model_dict.keys()]
    with open(f'models/model_{target[i]}_{model_name[2]}.pkl', 'rb') as f:
        model = pickle.load(f)
    std = joblib.load(f"standard_scaler_{target[i]}.pkl")
    X = pd.concat([features[best_feature[i]], ratio_all], axis=1)
    print(X)
    X_std = std.transform(X)
    Y_predict_MT = model.predict(X_std)

    i = 1
    model_name = [j for j in model_dict.keys()]
    with open(f'models/model_{target[i]}_{model_name[2]}.pkl', 'rb') as f:
        model = pickle.load(f)
    std = joblib.load(f"standard_scaler_{target[i]}.pkl")

    ratio_all.drop(['Ag', 'Cu', 'Ti'], axis=1)
    new_col = ['Sn', 'Bi', 'Cd', 'In', 'Pb', 'Zn', 'Ga', 'Al']
    ratio_all = ratio_all[new_col]
    X = pd.concat([features[best_feature[i]], ratio_all], axis=1)
    X_std = std.transform(X)
    Y_predict_enthalpy = model.predict(X_std)
    result = pd.DataFrame({"formula": features["formula"], "predict_MT": Y_predict_MT, "predict_enthalpy": Y_predict_enthalpy})
    result.to_csv(f"data/Ga_In_Sn_filter_{model_name[2]}.csv", index=False)