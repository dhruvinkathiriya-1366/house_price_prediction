import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from data_preprocessing import load_data,preprocess_data
from model import build_pipeine
from evaluate import eveluate_model


    
train_df,test_df=load_data()
test_id=test_df['Id']
train_df,test_df=preprocess_data(train_df,test_df)

x=train_df.drop(['SalePrice'],axis=1)
y=np.log1p(train_df['SalePrice'])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#detecting the columns
cat_col=train_df.select_dtypes(include="object").columns
ordinal_col=[
    "ExterQual", "ExterCond", "KitchenQual",
    "HeatingQC", "BsmtQual", "BsmtCond",
    "FireplaceQu", "GarageQual", "GarageCond"]  
nominal=[col for col in cat_col if col not in ordinal_col]
ordinal_order=["Po", "Fa", "TA", "Gd", "Ex"]

#train_model

pipeline=build_pipeine(nominal,ordinal_col,ordinal_order)
pipeline.fit(x_train,y_train)
train_r2,train_rmse=eveluate_model(pipeline,x_train,y_train)
print("train:R2 & Rmse")
print(f"R2:{train_r2}")
print(f"Rmse:{train_rmse}")  
test_R2,test_Rmse=eveluate_model(pipeline,x_test,y_test)
print("test:R2 & Rmse")
print(f"R2:{test_R2}")
print(f"Rmse:{test_Rmse}") 
# params = {
#     "n_estimators":[200,500,1000],
#     "learning_rate":[0.01,0.05, 0.1],
#     "max_depth":[3,4,5,6],
#     "subsample":[0.7,0.8,1.0],
#     "colsample_bytree":[0.7,0.8,1.0]
# }
# xgb =XGBRegressor(
#     random_state=42
# )

# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=params,
#     n_iter=20,
#     cv=5,
#     scoring="r2",
#     random_state=42,
#     n_jobs=-1,
#     verbose=2
# )
# random_search.fit(x_train,y_train)
# print(random_search.best_params_)
# print(random_search.best_score_)
