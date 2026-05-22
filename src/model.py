from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnFiller(BaseEstimator, TransformerMixin):
    def __init__(self, train_columns):
        self.train_columns = train_columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Add any missing columns with 0 / mode defaults
        for col in self.train_columns:
            if col not in X.columns:
                X[col] = 0
        return X[self.train_columns]
    
def build_pipeine(nominal,ordinal_col,ordinal_order,numeric_col):
    
    numeric_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
        )
    
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder",
        OrdinalEncoder(
            categories=[ordinal_order] * len(ordinal_col),
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
       ])
    nominal_transformer = Pipeline([
        
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])
   
    Preprocessor = ColumnTransformer([
        ("ord", ordinal_transformer, ordinal_col),
        ("nom", nominal_transformer, nominal),
        ("num", numeric_transformer, numeric_col)
        ])
    
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