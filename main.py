import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

df  = pd.read_csv("Housing.csv")
# print(df.head())

# print(df.isnull().sum())
# print(df.shape)
# print(df.duplicated().sum())
# print(df.dtypes)

#Boxplot
num_cols = df.select_dtypes(include=["int64"]).columns

for col in num_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    # plt.show()

binary_cols =["mainroad","guestroom","basement","hotwaterheating","airconditioning",
              "prefarea"]
df[binary_cols] = df[binary_cols].replace({"yes":1,"no":0})
df = pd.get_dummies(df, columns=["furnishingstatus"],drop_first=True)
# print(df.head())

target = 'price'

#Scatter plot
for col in df.columns:
    if col!= target:
        plt.figure()
        plt.scatter(df[col],df[target],alpha=0.3)
        plt.xlabel(col)
        plt.ylabel(target)
        plt.title(f"{col}vs{target}")
        # plt.show()

#Correlation before 
corr_matrix_b = df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix_b, annot=True, fmt=".2f",cmap="coolwarm")
plt.title("Correlation Matrix Before Preprocessing")
# plt.show()
df["price_log"] = np.log1p(df["price"])

X = df.drop(columns=['price_log'])
y = df['price_log']

#Split
X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.30,random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=42)

#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
x_test_scaled = scaler.transform(X_test)

df_scaled = pd.DataFrame(X_train_scaled, columns= X_train.columns)
# print(df_scaled.head())

#Evaluation Matrix
def evaluate(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr,y_tr)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    return rmse, r2

#Linear Regression
lr = LinearRegression()
rmse_lr, r2_lr = evaluate(
    lr, X_train_scaled, y_train, X_val_scaled, y_val
)
# print("Linear Regreession")
# print("RMSE: ", rmse_lr)
# print("R2: ", r2_lr)

#Ridge Regression
ridge = Ridge()
params = {
    "alpha":[0.01,0.1,1,10,100]
}

ridge_gs = GridSearchCV(
    ridge, params,
    scoring="neg_root_mean_squared_error",
    cv = 5
)

ridge_gs.fit(X_train_scaled, y_train)
best_ridge = ridge_gs.best_estimator_

rmse_ridge, r2_ridge = evaluate(
    best_ridge, X_train_scaled, y_train,X_val_scaled,y_val
)

print("Ridge Regreession")
print("RMSE: ", rmse_ridge)
print("R2: ", r2_ridge)



