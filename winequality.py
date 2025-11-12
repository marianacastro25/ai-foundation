'import data base'

import pandas as pd

df = pd.read_csv(r"C:\Users\maria\Documents\ai-foundation\winequality-red.csv", sep=';')
print(df.head())

'''
df.info()
print(df['quality'].value_counts().sort_index())

print(df.isnull().sum())
'''

print(df.nunique())

print(df.duplicated().sum())

df.drop_duplicates(inplace = True)
df = df.reset_index(drop=True)

num_cols = [col for col in df.columns if (df[col].dtype in ["int64","float64"]) & (df[col].nunique() > 50)]
print(num_cols)
target = [col for col in df.columns if df[col].nunique()<10]
print(target)


