# 定义合金元素的摩尔质量 (g/mol) 和密度 (g/cm³)
element_data = {
    "Ga": {"M": 69.723, "rho": 5.91},
    "Sn": {"M": 118.71, "rho": 7.31},
    "In": {"M": 114.82, "rho": 7.31},
    "Zn": {"M": 65.38, "rho": 7.14},
    "Cd": {"M": 112.41, "rho": 8.65},
    "Bi": {"M": 208.98, "rho": 9.78},
    "Pb": {"M": 207.2, "rho": 11.34}
}

# 所有合金成分，格式为列表（字典），角标为摩尔比
alloys = [
    {"Ga": 86.5, "Sn": 13.5},
    {"Ga": 100},
    {"Ga": 96.5, "Zn": 3.5},
    {"Ga": 82, "Sn": 12, "Zn": 6},
    {"Ga": 74, "Sn": 22, "Cd": 4},
    {"Ga": 93, "Zn": 5, "Cd": 2},
    {"Ga": 86, "Sn": 11, "Zn": 3},
    {"Ga": 67, "In": 20.5, "Sn": 12.5},
    {"Ga": 78.55, "In": 21.45},
    {"Ga": 75.5, "In": 24.5},
    {"Ga": 78, "In": 22},
    {"Bi": 49, "In": 51},
    {"Ga": 68, "In": 20, "Sn": 12},
    {"Ga": 68.5, "In": 21.5, "Sn": 10},
    {"Ga": 61, "In": 25, "Sn": 13, "Zn": 1},
    {"Sn": 100},
    {"In": 52.2, "Sn": 46, "Zn": 1.8},
    {"In": 52, "Sn": 48},
    {"Bi": 32.5, "In": 51, "Sn": 16.5},
    {"Bi": 32.5, "In": 51, "Sn": 16.5},
    {"Bi": 33.7, "In": 66.3},
    {"Bi": 30.8, "In": 61.7, "Cd": 7.5},
    {"Bi": 31.6, "In": 48.8, "Sn": 19.6},
    {"In": 51.34, "Sn": 5.56, "Bi": 33.1},
    {"Sn": 51.2, "Cd": 30.6, "Pb": 18.2},
    {"Bi": 0.355, "Sn": 0.601, "Zn": 0.044},
    {"Bi": 54, "In": 29.7, "Sn": 16.3},
    {"In": 4, "Sn": 40, "Bi": 56},
    {"Bi": 57, "In": 26, "Sn": 17},
    {"Bi": 58, "Sn": 42},
    {"In": 25.2, "Sn": 17.3, "Bi": 57.5},
    {"Sn": 26, "Bi": 53, "Cd": 21},
    {"Bi": 67, "In": 33},
    {"In": 10.5, "Sn": 19, "Bi": 53.5, "Pb": 17},
    {"Bi": 45, "Pb": 23, "In": 19, "Sn": 8, "Cd": 5},
    {"In": 21, "Sn": 12, "Bi": 49, "Pb": 18},
    {"In": 19.1, "Sn": 8.3, "Bi": 44.7, "Cd": 5.3, "Pb": 22.6},
    {"Sn": 22, "Bi": 50, "Pb": 28},
    {"Sn": 16, "Bi": 52, "Pb": 32},
    {"Sn": 13.3, "Bi": 50, "Cd": 10, "Pb": 26.7},
    {"Bi": 50, "Pb": 26.7, "Sn": 13.3, "Cd": 10},
    {"Sn": 15.5, "Bi": 52.5, "Pb": 32},
    {"Bi": 100},
    {"Bi": 51.6, "Cd": 8.2, "Pb": 40.2},
    {"Bi": 0.405, "Sn": 0.285, "Pb": 0.163, "In": 0.147}
]

# 计算密度的函数（质量分数法）
def compute_density(alloy):
    total_mass = sum(mol * element_data[el]["M"] for el, mol in alloy.items())
    mass_fractions = {el: mol * element_data[el]["M"] / total_mass for el, mol in alloy.items()}
    inverse_density = sum(w / element_data[el]["rho"] for el, w in mass_fractions.items())
    return round(1 / inverse_density, 3)

# 批量计算所有合金的密度
densities = [compute_density(alloy) for alloy in alloys]
print(densities)