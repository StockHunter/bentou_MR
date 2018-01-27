import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
from sklearn.linear_model import LinearRegression as LR

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample.csv",header=None)

y = train["y"]
trainX = train["week"].value_counts()

#dummy変数化
trainX  = pd.get_dummies(train[["week","temperature"]])

model = LR()
model.fit(trainX, y)

testX  = pd.get_dummies(test[["week", "temperature"]])

pred = model.predict(testX)
print(len(pred))

sample[1] = pred
sample.to_csv("submit2.csv", index=None, header=None)


