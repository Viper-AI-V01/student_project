import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

import dill

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys) # type: ignore
def evaluate_model(X_train,X_test,y_train,y_test,models,metrics):
    try:
        report ={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train) 
            y_train_pred = model.predict(X_train) 
            y_test_pred = model.predict(X_test) 
            train_model_score = metrics(y_train,y_train_pred)
            test_model_score = metrics(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info(f'Train model scores{model}:{train_model_score}')
            logging.info(f'Test model scores{model}:{test_model_score}')

        return report
    
    except Exception as e:
        logging.error(sys)
        raise CustomException(e,sys)# type: ignore
