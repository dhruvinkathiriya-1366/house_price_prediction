import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder 

# pd.set_option('display.max_columns',None)
# pd.set_option('display.width',None)
# pd.set_option('display.max_colwidth',None)
# pd.set_option('display.max_rows',None)


Train_path ="data/raw/train.csv"
Test_path ="data/raw/test.csv"

def load_data():
    train_df = pd.read_csv(Train_path)
    test_df = pd.read_csv(Test_path)
    # print("before preprocessing")
    # print(f"Train shape: {train_df.shape}")
    # print(f"Test shape: {test_df.shape}")

    return train_df, test_df

def preprocess_data(train_df, test_df):
    
    # Drop the low correlation column
    Drop_col=['Id','BsmtFinSF2','BsmtHalfBath','PoolArea','MoSold','3SsnPorch','MiscVal','LowQualFinSF','YrSold','OverallCond','MSSubClass']
    train_df.drop(columns=Drop_col,inplace=True)
    test_df.drop(columns=Drop_col,inplace=True)
    train_df['Alley']=train_df['Alley'].fillna('NoAlley')
    test_df['Alley']=test_df['Alley'].fillna('NoAlley')
    train_df =train_df.dropna(subset=['Electrical'])
    test_df=test_df.dropna(subset=['Electrical'])
    train_df.drop(['PoolQC', 'Fence', 'MiscFeature'],axis=1, inplace=True)
    test_df.drop(['PoolQC', 'Fence', 'MiscFeature'],axis=1, inplace=True)
    
    #fill the null value 
    train_df['LotFrontage'] = train_df.groupby('Neighborhood')['LotFrontage']\
                            .transform(lambda x: x.fillna(x.median()))
    test_df['LotFrontage'] = test_df.groupby('Neighborhood')['LotFrontage']\
                       .transform(lambda x: x.fillna(x.median()))
    
    # detecting the column 
    cat_col=train_df.select_dtypes(include="object").columns
    ordinal_col=[
    "ExterQual", "ExterCond", "KitchenQual",
    "HeatingQC", "BsmtQual", "BsmtCond",
    "FireplaceQu", "GarageQual", "GarageCond"]  
    nominal=[col for col in cat_col if col not in ordinal_col]
    
    # encoding
    ordinal_order=["Po", "Fa", "TA", "Gd", "Ex"]
    o_encoder=OrdinalEncoder(categories=[ordinal_order] * len(ordinal_col),handle_unknown="use_encoded_value",unknown_value=-1)
    encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    o_encoder_train=o_encoder.fit_transform(train_df[ordinal_col])
    o_encoder_test=o_encoder.transform(test_df[ordinal_col])

    encoder_train=encoder.fit_transform(train_df[nominal])
    encoder_test=encoder.transform(test_df[nominal])
    o_encoder_train_df=pd.DataFrame(
        o_encoder_train,
        columns=ordinal_col,
        index=train_df.index
    )
    o_encoder_test_df=pd.DataFrame(
        o_encoder_test,
        columns=ordinal_col,
        index=test_df.index
    )
    
    encod_train_df = pd.DataFrame(
     encoder_train,
     columns=encoder.get_feature_names_out(nominal),
     index=train_df.index
     ) 
    encod_test_df = pd.DataFrame(
    encoder_test,
    columns=encoder.get_feature_names_out(nominal),
    index=test_df.index
    )
    train_df = train_df.drop(columns=nominal)
    test_df = test_df.drop(columns=nominal)
    train_df=train_df.drop(columns=ordinal_col)
    test_df=test_df.drop(columns=ordinal_col)
    
    train_df=pd.concat([train_df,o_encoder_train_df],axis=1)
    test_df=pd.concat([test_df,o_encoder_test_df],axis=1)
    train_df = pd.concat([train_df, encod_train_df],axis=1)
    test_df = pd.concat([test_df, encod_test_df], axis=1)
    # print("after preprocessing")
    # print(f"Train shape: {train_df.shape}")
    # print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data() 
    
    train_df, test_df = preprocess_data(train_df, test_df) 