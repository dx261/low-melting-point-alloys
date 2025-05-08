from util.descriptor import magpie
import pandas as pd
import numpy as np
import os
from util.alloys_features import normalize_element_dict, find_elements
from util.descriptor.magpie import get_magpie_features

if __name__ == '__main__':
    target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    i = 0
    df = pd.read_csv(f"data/{target[i]}.csv")
    df = df.drop_duplicates(keep="first", inplace=False)
    data_path, file_name = "data", f"{target[i]}_formula.csv"
    formula_path = os.path.join(data_path, file_name)
    pd.DataFrame(df["formula"], columns=["formula"]).to_csv(formula_path, index=False)
    df_magpie = get_magpie_features(file_name=f"{target[i]}_formula.csv", data_path="data", alloy_features=True)
    data = pd.concat([df[f"{target[i]}"], df_magpie], axis=1)
    data.to_csv(f"data/magpie_{target[i]}.csv", index=False)