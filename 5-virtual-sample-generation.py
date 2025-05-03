import itertools
import numpy as np
import os
import pandas as pd
import torch
from util.deep_learning.VAE.WAE import WAETrainer
from util.deep_learning.VAE.base import OneDimensionalDataset
from util.alloys_features import normalize_element_dict, find_elements

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

def formula_to_ratio_dataset(dataset):
    """
    :param dataset: contain the formula column with alloys formula
    :return:
    """
    df_ratio = pd.DataFrame()
    if not "formula" in list(dataset.columns):
        raise AssertionError(f"Error: no column named formula is not in the dataset !")
    for i, formula in enumerate(dataset.formula):
        element_dict = find_elements(formula)
        normalized = normalize_element_dict(element_dict)
        # print(normalized)
        df_formula = pd.DataFrame(data=normalized, index=[i])
        df_ratio = pd.concat([df_ratio, df_formula])
    df_ratio = df_ratio.fillna(0).reset_index(drop=True)
    dataset_reindex = dataset.reset_index(drop=True)
    df_all = pd.concat([dataset_reindex, df_ratio], axis=1)
    element_columns = list(df_ratio.columns)
    return df_all, element_columns

if __name__ == '__main__':
    # # 1-高通量方法
    # # generate virtual space
    # # 键是元素，值是比例列表
    # search_range = {"Sn": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 "Bi": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 "In": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 "Ga": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 "Zn": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 "Ti": [i / 100 for i in range(0, 101, 10)],  # 范围0%-100%
    #                 }
    # uniques = [i for i in search_range.values()]  # 这行代码提取了所有元素的摩尔比例列表，形成一个列表 uniques
    # all_element_ratios = []
    # for element_ratio in itertools.product(*uniques):  # 做笛卡尔积：枚举所有元素可能摩尔比例的组合
    #     if 1 < element_ratio.count(0) < 5:  # 控制每个样本的元素种类在2到4个之间
    #         all_element_ratios.append(element_ratio)
    # result = pd.DataFrame(all_element_ratios, columns=list(search_range.keys()))
    # result.to_csv("./data/virtual_samples_lmp_alloys.csv", index=False)
    # # 比例式转换为化学式
    # df = pd.read_csv("./data/virtual_samples_lmp_alloys.csv")
    # formula = get_chemical_formula(df)
    # pd.DataFrame(formula).to_csv("./data/virtual_samples_lmp_alloys_formula.csv", index=False)

    # # 2-WAE/VAE等根据原数据分布的生成算法
    # target = ["melting_point", "enthalpy", "enthalpy-J-cc", "density"]
    # i = 0
    # dataset = pd.read_csv(f"data/{target[i]}.csv")
    # dataset, b = formula_to_ratio_dataset(dataset)
    # dataset.to_csv(f"data/{target[i]}_ratio.csv", index=False)
    #
    # elements_col = list(dataset.columns[2:])
    # df_element = dataset[elements_col]
    # # 计算合金的组分个数
    # df_element["N_alloy"] = df_element.astype(bool).sum(axis=1)
    # print(df_element.head())
    # input_dim = df_element.shape[1]  # 获得列的个数
    # print(input_dim)
    #
    # data = torch.Tensor(df_element.values)
    #
    # # 创建 WAE 训练器实例
    # print("WAE training")
    # trainer = WAETrainer(OneDimensionalDataset(data), input_dim=input_dim)
    # # 训练模型
    # trainer.train(epochs=200)
    # print("WAE training finished")
    # print("WAE generation")
    # # 生成样本
    # gen = True
    # if gen:
    #     generated_samples_scaled = trainer.generate_samples(num_samples=10000).numpy()
    #     # 反归一化得到生成样本 (成分 + 元素个数）
    #     generated_samples = trainer.data.scaler.inverse_transform(generated_samples_scaled)
    #     df_gen_source = pd.DataFrame(generated_samples_scaled, columns=df_element.columns)
    #     # 去掉N_alloy列
    #     df_gen_source.drop(columns=["N_alloy"], inplace=True)
    #     df_gen = pd.DataFrame(generated_samples, columns=df_element.columns)
    #     N_alloy = -1 * df_gen["N_alloy"].round().astype(int).copy()
    #     # print(N_alloy.head())
    #     # print(df_gen.head())
    #     # print(df_gen_source.head())
    #     # 每一行 找到前n大的元素
    #     for index, row in df_gen_source.iterrows():
    #         n = N_alloy[index] * -1  # 找前几个最大元素
    #         top_n_indices = row.nlargest(n).index
    #         new_row = [0] * len(row)
    #         for col in top_n_indices:
    #             new_row[df_gen_source.columns.get_loc(col)] = row[col]
    #         df_gen_source.loc[index] = new_row
    #     df_gen_source["N_alloy"] = N_alloy
    #     # 归一化
    #     gen = trainer.data.scaler.inverse_transform(df_gen_source.values)
    #     df_gen = pd.DataFrame(gen, columns=df_element.columns)
    #     # 归一化所有元素和为100
    #     df_gen = df_gen[elements_col].div(df_gen[elements_col].sum(axis=1), axis=0) * 100
    #     # save
    #     df_gen.to_csv('data/generated_samples_WAE.csv', index=False)

    # 3-WAE生成的样本合并成化学式
    df = pd.read_csv("data/generated_samples_WAE.csv")
    formula = get_chemical_formula(df)
    pd.DataFrame(formula).to_csv("data/generated_samples_WAE_formula.csv", index=False)