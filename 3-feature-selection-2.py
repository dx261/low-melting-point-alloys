from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import ExtraTreesRegressor
from time import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE
import numpy as np
from sklearn.model_selection import cross_val_score as CVS
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

if __name__ == '__main__':
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    i = 1
    data_path, file_name = "data", f"magpie_{target[i]}_feature_selected-1.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    print(df.shape)
    df.drop_duplicates(keep="first", inplace=True)
    df.dropna(inplace=True)
    print(df.shape)
    features = [i for i in df.columns if i not in target]
    X = df[features]
    Y = df[target[i]]

    # 数据划分与标准化
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
    std = StandardScaler()
    X_train_std = std.fit_transform(X_train)
    X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
    X_test_std = std.transform(X_test)
    X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)

    model = ExtraTreesRegressor(random_state=1).fit(X_train_std, Y_train)
    select_feature_all_times, avg_mse = [], []
    tic_fwd = time()

    sfs_forward = SFS(model, k_features=len(features), forward=True,
                      cv=10, scoring='neg_mean_squared_error')
    sfs_forward.fit(X_train_std, Y_train)

    sfs_results = sfs_forward.subsets_
    for i in range(len(features)):
        select_feature_all_times.append(sfs_results[i + 1]['feature_names'])
        avg_mse.append(sfs_results[i + 1]['avg_score'])
    toc_fwd = time()
    avg_mse = [-1 * i for i in avg_mse]
    avg_rmse = [np.sqrt(i) for i in avg_mse]
    print(f"Done in {toc_fwd - tic_fwd:.3f}s")
    print('SFS RMSE  ---> ', avg_rmse, '\n特征名  ---> ', select_feature_all_times)
    x = range(1, len(features) + 1)

    # alpha 0表示完全透明，1表示完全不透明
    plt.plot(x, avg_rmse, linestyle=':', marker='8', color='#65ab7c', markerfacecolor='#0d75f8',
             markeredgecolor='#0d75f8',
             alpha=0.5, linewidth=1, ms=6)  # 原本颜色：#1a6fdf+#f14040
    # plt.legend() # 让图例生效
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(avg_rmse) - 0.5, max(avg_rmse) + 0.5)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"Number of features")  # X轴标签
    plt.ylabel("RMSE")  # Y轴标签
    min_indx = np.argmin(avg_rmse)  # min value index
    # 打印最佳特征子集
    best_features = select_feature_all_times[min_indx]
    print('SFS svr 最佳特征子集-------> ', best_features, '\n最佳特征子集数量-------> ', min_indx + 1)
    # 标记最低点
    plt.plot(x[min_indx], avg_rmse[min_indx], marker='*', markerfacecolor='r', markeredgecolor='r',
             alpha=0.5, ms=10)
    show_min = '[' + str(x[min_indx]) + ',' + str(round(avg_rmse[min_indx], 2)) + ']'
    plt.annotate(show_min, xytext=(x[min_indx], avg_rmse[min_indx] + 0.2), xy=(x[min_indx], avg_rmse[min_indx]))
    # 上一行标记最低点的具体数值
    plt.show()

