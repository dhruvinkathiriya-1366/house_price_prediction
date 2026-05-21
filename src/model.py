from sklearn.ensemble import RandomForestRegressor

def train_model(x_train,y_train):
    
    model=RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42)
    model.fit(x_train,y_train)
    
    return model