from util.descriptor import magpie
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    # # 1-使用高通量生成的样本
    # df = pd.read_csv(f"data/virtual_samples_lmp_alloys_formula.csv")
    df = pd.read_csv(f"data/virtual_samples_lmp_alloys_formula_Ga_In_Sn.csv")
    # 2-使用WAE生成的样本
    num_samples = 50000
    # df = pd.read_csv(f"data/generated_samples_WAE_{num_samples}_formula.csv")
    # df_test = df.iloc[17000:18000, :]
    # df_test.to_csv(f"data/virtual_samples_test.csv", index=False)

    WAE = False
    specific_elements = True
    if WAE:
        df_magpie = magpie.get_magpie_features(file_name=f"generated_samples_WAE_{num_samples}_formula.csv", data_path="data", alloy_features=True)
        df_magpie.to_csv(f"data/magpie_virtual_sample_WAE_{num_samples}.csv", index=False)
    elif specific_elements:
        df_magpie = magpie.get_magpie_features(file_name=f"virtual_samples_lmp_alloys_formula_Ga_In_Sn.csv", data_path="data", alloy_features=True)
        df_magpie.to_csv(f"data/magpie_virtual_sample_Ga_In_Sn.csv", index=False)
    else:
        df_magpie = magpie.get_magpie_features(file_name=f"generated_samples_formula.csv",
                                               data_path="data", alloy_features=True)
        df_magpie.to_csv(f"data/magpie_virtual_sample.csv", index=False)