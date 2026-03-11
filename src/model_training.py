# 1. Load processed data from processed folder
# 2. Create ML model and train
# 3. Save model artifact folder

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
x_train=pd.read_csv("../data/processed/x_train.csv")
x_test=pd.read_csv("../data/processed/x_test.csv")
y_train=pd.read_csv("../data/processed/y_train.csv")
y_test=pd.read_csv("../data/processed/y_test.csv")

print(x_train)

model=LinearRegression()
model.fit(x_train,y_train)

with open("../artifacts/model.pkl","wb") as f:
    pickle.dump(model,f)

