import pandas as pd
import numpy as np
import re
from util.base_function import get_chemical_formula

def normalize_formula(formula):
    # 匹配元素符号和数值（如 Sn0.5）
    pairs = re.findall(r'([A-Z][a-z]*)([0-9.]+)', formula)
    elements = []
    amounts = []
    for el, amt in pairs:
        elements.append(el)
        amounts.append(float(amt))
    total = sum(amounts)
    normalized = {el: round(amt / total, 2) for el, amt in zip(elements, amounts)}
    return normalized

if __name__ == '__main__':
    df = pd.read_csv(f'data/Ga_In_Sn_filter_ridge.csv')
    formulas = list(df['formula'])
    normalized_data = [normalize_formula(f) for f in formulas]
    data = pd.DataFrame(normalized_data).fillna(0)
    df_all = pd.concat([df, data], axis=1)
    for index, row in df_all.iterrows():
        if row['Sn'] + row['In'] + row['Ga'] != 1:
            df_all.drop(index, inplace=True)
    df_all = df_all.drop_duplicates(subset=['Sn', 'In', 'Ga'], keep='first', inplace=False)
    formula_atom_fraction = get_chemical_formula(df_all[['Sn', 'In', 'Ga']])
    df_all['formula_atom_fraction'] = [formula for formula in formula_atom_fraction]
    df_all.to_csv(f'data/Ga_In_Sn_filter_ridge_2.csv', index=False)