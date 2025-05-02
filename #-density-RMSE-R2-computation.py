import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


df = pd.read_csv("data/density_compute.csv")
a = df["density_real"]
b = df["density_computation"]
print(r2_score(a, b), pearsonr(a, b), np.sqrt(mean_squared_error(a, b)))
# 经验公式可以很好地计算合金密度
