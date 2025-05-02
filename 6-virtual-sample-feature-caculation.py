from util.descriptor import magpie
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    df = pd.read_csv(f"data/virtual_samples_lmp_alloys_formula.csv")
    df_magpie = magpie.get_magpie_features(file_name=f"virtual_samples_lmp_alloys_formula.csv", data_path="data", alloy_features=True)
    df_magpie.to_csv(f"data/magpie_virtual_sample.csv", index=False)