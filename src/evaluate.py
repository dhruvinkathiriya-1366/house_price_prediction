from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def eveluate_model(model,x_test,y_test):
    y_pred=model.predict(x_test)
    y_pred=np.expm1(y_pred)
    y_test=np.expm1(y_test)
    
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)
    
    return r2,rmse