from util.descriptor import magpie
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    df = pd.read_csv("data/wae50000_filter.csv")
    df["formula"].to_csv(f"data/wae50000_filter_formula.csv", index=False)
    df_magpie = magpie.get_magpie_features(file_name=f"wae50000_filter_formula.csv", data_path="data", alloy_features=True)
    df_magpie.to_csv(f"data/magpie_wae50000_filter.csv", index=False)