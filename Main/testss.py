import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# l = [1,2 ,3]
# lists = [l]
# xd = lists[0]
# print(xd[-1])

df = pd.DataFrame({'xd':[1,2,3,4,5], 'xdd': ['a', 'b', 'c', 'd', 'f']})
print(df['xd'].unique())