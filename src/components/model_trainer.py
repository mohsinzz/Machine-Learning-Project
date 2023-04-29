import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,model_evaluator

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join('artifacts','modeltrainer.pkl')

class ModelTrainer:
    def __init__(self) :
        self.model_config = ModelTrainerConfig()
    def initiate_model_training(self,salary_train_arr,salary_test_arr):
        try:
            train_data = salary_train_arr
            #pd.read_csv('C:\Machine Learning Project\artifacts\salary_train.csv')
            test_data = salary_test_arr
            #pd.read_csv('C:\Machine Learning Project\artifacts\salary_test.csv')


            X_train,y_train,X_test,y_test = train_data[:,:-1],train_data[:,-1],test_data[:,:-1],test_data[:,-1]

            models  = {"Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "AdaBoost Regressor": AdaBoostRegressor()
                }
            params={
                    "Decision Tree": {
                        'criterion':['squared_error', 'friedman_mse'],
                        'splitter':['best','random'],
                        'max_features':['sqrt','log2'],
                        'random_state' : [30,40,50],
                        'max_depth' : [5,6,7,8],
                        'max_features' : [6,8,10,12]
                    },
                    "Random Forest":{
                        'criterion':['squared_error', 'friedman_mse'],
                    
                        'max_features':['sqrt','log2'],
                        'n_estimators': [8,16,32,64]
                    },
                    "Gradient Boosting":{
                        #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.8,0.85,0.9],
                        'criterion':['squared_error', 'friedman_mse'],
                        'max_features':['sqrt','log2'],
                        'n_estimators': [8,16,32,64]
                    },
                    "Linear Regression":{},
                    "XGBRegressor":{'booster':['gblinear'],
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,100]
                    },
                    
                    "AdaBoost Regressor":{
                        'learning_rate':[.1,0.5],
                        # 'loss':['linear','square','exponential'],
                        'n_estimators': [8,16,32,64]
                    }
                    
                }
            
            logging.info('Obtaining the best model score')

            bestr2_models = model_evaluator(X1=X_train,y1=y_train,X2=X_test,y2=y_test,models_dict=models,params_dict=params)

            logging.info('Obtaining the best model name')

            best_model_score = max(sorted(bestr2_models.values()))
            best_model_name = list(bestr2_models.keys())[list(bestr2_models.values()).index(best_model_score)]

            best_model = models[best_model_name]

            save_object(file_path=self.model_config.model_trainer_file_path,obj= best_model)

            return (best_model_name,best_model_score)
        except Exception as e:
            raise CustomException(e,sys)
        





