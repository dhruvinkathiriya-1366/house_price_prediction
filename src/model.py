from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

def build_pipeine(nominal,ordinal_col,ordinal_order):
    
    Preprocessor=ColumnTransformer(transformers=[
        ('cat',OrdinalEncoder(categories=[ordinal_order] * len(ordinal_col),handle_unknown="use_encoded_value",unknown_value=-1),ordinal_col),
        ('nom',OneHotEncoder(handle_unknown="ignore", sparse_output=False),nominal)
        ],remainder="passthrough")
    
    model=XGBRegressor(
        n_estimators=200,
        colsample_bytree=0.7,
        learning_rate=0.05, 
        max_depth=3,
        subsample=0.7,
        random_state=42,
        n_jobs=-1
        )
    pipeline=Pipeline([
        ("preprocessor",Preprocessor),
        ("model",model)
    ])
    
    return pipeline