import itertools
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_chemical_formula(dataset):
    """
    Al   Ni   Si
    0.5  0.5  0
    :return: get_chemical_formula from element mol weigh dataframe Al0.5Ni0.5
    """
    elements_columns = dataset.columns
    dataset = dataset.reset_index()
    chemistry_formula = []
    for i in range(dataset.shape[0]):
        single_formula = []
        for col in elements_columns:
            if (dataset.at[i, col]) > 0:
                # element
                single_formula.append(col)
                # ratio
                single_formula.append(str(dataset.at[i, col]))
        chemistry_formula.append("".join(single_formula))
    return chemistry_formula

if __name__ == '__main__':
    # # 1-高通量方法
    # # generate virtual space
    # # 一个键是元素，值是比例列表的字典
    search_range = {"Sn": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    "Bi": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    "In": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    "Ga": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    "Zn": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    "Ti": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
                    }
    uniques = [i for i in search_range.values()]  # 这行代码提取了所有元素的摩尔比例列表，形成一个列表 uniques
    all_element_ratios = []
    for element_ratio in itertools.product(*uniques):  # 做笛卡尔积：枚举所有元素可能摩尔比例的组合
        if 1 < element_ratio.count(0) < 5:
            all_element_ratios.append(element_ratio)
    result = pd.DataFrame(all_element_ratios, columns=list(search_range.keys()))
    result.to_csv("./data/virtual_samples_lmp_alloys.csv", index=False)
    # 比例式转换为化学式
    df = pd.read_csv("./data/virtual_samples_lmp_alloys.csv")
    formula = get_chemical_formula(df)
    pd.DataFrame(formula).to_csv("./data/virtual_samples_lmp_alloys_formula.csv", index=False)

    # 2-WAE/VAE等根据原数据分布的生成算法
