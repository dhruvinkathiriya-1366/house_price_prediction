import pandas as pd


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
    
    print("after preprocessing")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data() 
    
    train_df, test_df = preprocess_data(train_df, test_df) 