from pymatgen.core import Composition, Element
import pandas as pd

df = pd.read_csv("data/virtual_samples_lmp_alloys_formula.csv")
bad_rows = []

for idx, formula in enumerate(df['formula']):
    try:
        comp = Composition(formula)
        for el in comp.elements:
            _ = Element(el.symbol)  # 如果不是合法元素，这里会报错
    except Exception as e:
        print(f"❌ Row {idx}: '{formula}' -> {e}")
        bad_rows.append((idx, formula))