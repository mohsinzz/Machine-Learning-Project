import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import dill
from src.logger import logging

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
#creating the model 
   
def model_evaluator(X1,y1,X2,y2,models_dict,params_dict):
    try:
        model_best_r2score={}
        for i in range(len(list(models_dict.values()))):

            model_obj = list(models_dict.values())[i]
            model_params = list(params_dict.values())[i]

            grid = GridSearchCV(model_obj,model_params,cv=3,refit = True, verbose = 3)

            grid.fit(X1,y1)

            model_obj.set_params(**grid.best_params_)
            model_obj.fit(X1,y1)

            y_pred = model_obj.predict(X2)
            

            model_best_r2score[list(models_dict.keys())[i]]=grid.score(X2,y2)
            logging.info(list(models_dict.keys())[i],'A part of model training is completed')

        return model_best_r2score
    
        logging.info('All the models have been evaluated')
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:

        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:

        raise CustomException(e,sys)



    




    





