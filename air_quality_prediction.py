import pandas as pd
from sklearn.model_selection import train_test_split


df=pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

print(df.info())

df=df.drop(['No'],axis=1)

df=df.dropna()

y=df['pm2.5']
X=df.drop(['pm2.5'],axis=1)

train_X,test_X,train_y,test_y=train_test_split(X,y)

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

le=LabelEncoder()

train_cbwd=le.fit_transform(train_X['cbwd'])
test_cbwd=le.transform(test_X['cbwd'])

train_cbwd=train_cbwd.reshape(len(train_cbwd),1)
test_cbwd=test_cbwd.reshape(-1,1)
ohe=OneHotEncoder(sparse=False)

train_encoded=ohe.fit_transform(train_cbwd)
test_encoded=ohe.transform(test_cbwd)

train_encoded=pd.DataFrame(train_encoded).reset_index()
test_encoded=pd.DataFrame(test_encoded).reset_index()

train_X=train_X.drop(['cbwd'],axis=1)
test_X=test_X.drop(['cbwd'],axis=1)

train_X=train_X.reset_index()
test_X=test_X.reset_index()

train_X=pd.concat([train_encoded,train_X],axis=1)
test_X=pd.concat([test_encoded,test_X],axis=1)

train_X.drop(['index'],axis=1,inplace=True)
test_X.drop(['index'],axis=1,inplace=True)

from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(max_leaf_nodes=100)

model.fit(train_X,train_y)


output=model.predict(train_X)
print(output)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(train_y,output))
print(mean_squared_error(test_y,model.predict(test_X)))

import xgboost as xgb

model=xgb.XGBRegressor()

model.fit(train_X,train_y)


output=model.predict(train_X)
print(output)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(train_y,output))
print(mean_squared_error(test_y,model.predict(test_X)))