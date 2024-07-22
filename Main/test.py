import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df_test1 = pd.read_csv('test_FD001.txt')
df_train1 = pd.read_csv('train_FD001.txt')

print(df_test1.index)