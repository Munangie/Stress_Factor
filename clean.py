import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
df = pd.read_csv('stress_factor.csv')
corr = df.corr()
X = df[['study load', 'sleep quality']]
y = df['stress levels']
model = LinearRegression().fit(X, y)
print(model.coef_)
