# 1. Load raw Data
# 2. Identifying x and y (input or output)
# 3. Split data into train and test
import pandas as pd
from sklearn.model_selection import train_test_split
def load_and_split():
    df=pd.read_csv(r"C:\Users\adity\Desktop\Tekworks\Week-8\Day-15\Projects\Insurance_prediction\data\raw\insurance_data.csv")
    x=df[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=df['Annual_Premium_Thousands']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    return x_train,x_test,y_train,y_test