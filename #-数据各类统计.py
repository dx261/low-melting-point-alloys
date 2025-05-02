import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    i = 0
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    data_path, file_name = "data", f"magpie_{target[i]}_feature_selected-1.csv"
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)