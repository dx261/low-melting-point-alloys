import pandas as pd
import re
from collections import Counter

# 读取数据文件
df = pd.read_csv('data/melting_point_formula.csv')

# 用于提取元素的函数
def extract_elements(formula):
    # 使用正则表达式匹配元素符号（大写字母后跟可选的小写字母）
    elements = re.findall(r'[A-Z][a-z]*', formula)
    return elements

# 统计所有元素
all_elements = []
for formula in df['formula']:
    elements = extract_elements(formula)
    all_elements.extend(elements)

# 统计每个元素出现的次数
element_counts = Counter(all_elements)

# 打印结果
print("出现的元素及其出现次数：")
for element, count in element_counts.most_common():
    print(f"{element}: {count}次") 