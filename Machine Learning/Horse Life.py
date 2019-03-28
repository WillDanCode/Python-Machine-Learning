import numpy as np
import pandas as pd

df = pd.read_csv('../Dataset/horse.csv')

print(df.isnull().sum())
print('\n')
print(df['rectal_temp'].plot())