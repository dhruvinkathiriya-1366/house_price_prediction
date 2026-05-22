from xgboost import XGBRegressor

def train_model(x_train,y_train):
    
    model=XGBRegressor(
        n_estimators=200,
        colsample_bytree=0.7,
        learning_rate=0.05, 
        max_depth=3,
        subsample=0.7,
        random_state=42,
        n_jobs=-1
        )
    model.fit(x_train,y_train)
    
    return model