import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data,preprocess_data
from model import train_model
from evaluate import eveluate_model


    
train_df,test_df=load_data()
test_id=test_df['Id']
train_df,test_df=preprocess_data(train_df,test_df)

x=train_df.drop(['SalePrice'],axis=1)
y=np.log1p(train_df['SalePrice'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=train_model(x_train,y_train)
train_r2,train_rmse=eveluate_model(model,x_train,y_train)
#print("train:R2 & Rmse")
#print(f"R2:{train_r2}")
#print(f"Rmse:{train_rmse}")  
test_R2,test_Rmse=eveluate_model(model,x_test,y_test)
#print("test:R2 & Rmse")
#print(f"R2:{test_R2}")
#print(f"Rmse:{test_Rmse}") 

#save the model
joblib.dump(model,"models/houce_price_model.pkl")
model=joblib.load("models/houce_price_model.pkl")

#predict
final_pred = np.expm1(model.predict(test_df))
print(final_pred)

#create the submission.csv

Submission=pd.DataFrame({
    "Id":test_id,
    "SalePrice":final_pred
})

Submission.to_csv(
    "data/processed/Submission.csv",
    index=False
)
