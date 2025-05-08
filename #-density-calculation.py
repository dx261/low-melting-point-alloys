from typing import List, Tuple
import re
import pandas as pd

# 密度 (g/cm³)，常温下近似值
density_data = {
    "Sn": 7.31,
    "In": 7.31,
    "Ga": 5.91,
}

# 相对原子质量（g/mol）
atomic_mass = {
    "Sn": 118.71,
    "In": 114.82,
    "Ga": 69.72,
}


# 提取合金字符串的元素及摩尔比
def parse_formula(formula: str) -> List[Tuple[str, float]]:
    matches = re.findall(r'(Sn|In|Ga)(\d*\.?\d+)', formula)
    return [(elem, float(ratio)) for elem, ratio in matches]


# 质量分数法计算合金密度
def calculate_density(formula: str) -> float:
    components = parse_formula(formula)

    total_mass = sum(atomic_mass[elem] * ratio for elem, ratio in components)

    mass_fractions = [(elem, (atomic_mass[elem] * ratio) / total_mass) for elem, ratio in components]

    density = sum(mf / density_data[elem] for elem, mf in mass_fractions) ** -1

    return round(density, 4)


# 提供的配方列表
formulas = '''
Sn0.23In0.46Ga0.31
Sn0.05In0.68Ga0.27
In0.71Ga0.29
Sn0.19Ga0.81
Sn0.24In0.39Ga0.37
Sn0.22Ga0.78
Sn0.22In0.5Ga0.28
Sn0.24In0.4Ga0.36
Sn0.26In0.36Ga0.38
Sn0.23Ga0.77
Sn0.24In0.38Ga0.38
Sn0.28In0.21Ga0.51
Sn0.23In0.47Ga0.3
Sn0.09In0.65Ga0.26
Sn0.29In0.2Ga0.51
Sn0.19In0.53Ga0.28
Sn0.2In0.52Ga0.28
Sn0.21Ga0.79
Sn0.11In0.63Ga0.26
Sn0.24In0.42Ga0.34
Sn0.29In0.19Ga0.52
Sn0.27In0.31Ga0.42
Sn0.28In0.22Ga0.5
Sn0.26In0.35Ga0.39
Sn0.28In0.25Ga0.47
Sn0.19In0.54Ga0.27
Sn0.18In0.55Ga0.27
Sn0.13In0.61Ga0.26
Sn0.28In0.24Ga0.48
Sn0.22In0.51Ga0.27
Sn0.23In0.48Ga0.29
Sn0.04In0.7Ga0.26
Sn0.16In0.58Ga0.26
Sn0.15In0.59Ga0.26
Sn0.24In0.47Ga0.29
Sn0.25In0.38Ga0.37
Sn0.25In0.39Ga0.36
Sn0.21In0.52Ga0.27
Sn0.2In0.53Ga0.27
Sn0.27In0.29Ga0.44
Sn0.27In0.3Ga0.43
Sn0.29In0.12Ga0.59
Sn0.3In0.13Ga0.57
Sn0.3In0.12Ga0.58
In0.72Ga0.28
Sn0.27In0.33Ga0.4
Sn0.28In0.26Ga0.46
Sn0.07In0.67Ga0.26
Sn0.29In0.18Ga0.53
Sn0.29In0.21Ga0.5
Sn0.3In0.15Ga0.55
Sn0.29In0.15Ga0.56
Sn0.29In0.25Ga0.46
Sn0.26In0.38Ga0.36
Sn0.3In0.16Ga0.54
Sn0.28In0.28Ga0.44
Sn0.25In0.4Ga0.35
Sn0.23In0.49Ga0.28
Sn0.26In0.37Ga0.37
Sn0.17In0.57Ga0.26
Sn0.18In0.56Ga0.26
Sn0.3In0.11Ga0.59
Sn0.24Ga0.76
Sn0.25In0.41Ga0.34
Sn0.28In0.23Ga0.49
Sn0.29In0.24Ga0.47
In0.73Ga0.27
Sn0.29In0.1Ga0.61
Sn0.29In0.26Ga0.45
Sn0.27In0.08Ga0.65
Sn0.28In0.08Ga0.64
Sn0.3In0.19Ga0.51
Sn0.21In0.53Ga0.26
Sn0.29In0.23Ga0.48
Sn0.22In0.52Ga0.26
Sn0.3In0.17Ga0.53
'''.strip().splitlines()  # 按行分割字符串

# 批量计算密度
density_results = [(f, calculate_density(f)) for f in formulas]
df = pd.DataFrame(density_results, columns=["Formula", "Density (g/cm3)"])
df.head(10)  # 显示前10行结果作为示例
df.to_csv("data/virtual_sample_density_results.csv", index=False)