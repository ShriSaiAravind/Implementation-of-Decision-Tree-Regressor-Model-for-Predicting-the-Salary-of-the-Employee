import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv("Salary.csv")
print(data.head())
print(f"\n{data.info()}")
print(f"\n{data.isnull().sum()}")


le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


dt=DecisionTreeRegressor().fit(x_train.values,y_train.values)
y_pred=dt.predict(x_test.values)


mse=mean_squared_error(y_test,y_pred)
print(mse)

r2=r2_score(y_test,y_pred)
print(r2)

print(dt.predict([[5,6]]))