import pandas as pd
import numpy as np
import re

if __name__=='__main__':
    df = pd.read_csv('data/final_formula.csv')

    def normalize_formula(formula):
        # 匹配元素符号和数值（如 Sn0.5）
        pairs = re.findall(r'([A-Z][a-z]*)([0-9.]+)', formula)
        elements = []
        amounts = []
        for el, amt in pairs:
            elements.append(el)
            amounts.append(float(amt))
        total = sum(amounts)
        normalized = {el: amt / total for el, amt in zip(elements, amounts)}
        return normalized

    # 化学式列表
    formulas = list(df['formula'])

    # 转换为 DataFrame
    normalized_data = [normalize_formula(f) for f in formulas]
    df = pd.DataFrame(normalized_data).fillna(0)
    df.round(2).to_csv('data/final_formula_guiyi.csv')

    # 修改

    aftar_revise = pd.read_csv('data/final_formula_guiyi.csv')
    print(aftar_revise)
    aftar_revise = aftar_revise.drop_duplicates(subset=['Sn', 'In', 'Ga'], keep='first', inplace=False)
    for index, row in aftar_revise.iterrows():
        print(row)
        if row['Sn'] + row['In'] + row['Ga'] != 1:
            aftar_revise.drop(index, inplace=True)
    aftar_revise.to_csv('data/final_formula_guiyi.csv')