import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df  = pd.read_csv("Housing.csv")
# print(df.head())

print(df.isnull().sum())
print(df.shape)
print(df.duplicated().sum())
print(df.dtypes)

#Boxplot
num_cols = df.select_dtypes(include=["int64"]).columns

for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()

binary_cols =["mainroad","guestroom","basement","hotwaterheating","airconditioning",
              "prefarea"]
df[binary_cols] = df[binary_cols].replace({"yes":1,"no":0})
df = pd.get_dummies(df, columns=["furnishingstatus"],drop_first=True)
print(df.head())

target = 'price'

#Scatter plot
for col in df.columns:
    if col!= target:
        plt.figure()
        plt.scatter(df[col],df[target],alpha=0.3)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col}vs{target}")
        plt.show()

#Correlation before 
corr_matrix_b = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix_b, annot=True, fmt=".2f",cmap="coolwarm")
plt.title("Correlation Matrix Before Preprocessing")
plt.show()

X = df.drop(columns=['price'])
y = df['price']

#Split
X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.30,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=42)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
x_test_scaled = scaler.transform(X_test)

df_scaled = pd.DataFrame(X_train_scaled, columns= X_train.columns)
print(df_scaled.head())


