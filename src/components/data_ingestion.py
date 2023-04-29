import os
import sys
from src.logger import logging
import pandas as pd
from src.exception import CustomException
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

from src.components.data_transformation import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class Dataingestionconfig:
    train_data_path :str =os.path.join('artifacts','salary_train.csv')
    test_data_path  :str =os.path.join('artifacts','salary_test.csv')
    raw_data_path   :str =os.path.join('artifacts','salary_data.csv')
class DataIngestion:
    def __init__(self) :
        self.ingestion_config = Dataingestionconfig()
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')

        try:
            salary = pd.read_csv('data\data_final.csv')
            logging.info('Read the data as a Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            salary.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train-Test split has been initiated')
            train_set,test_set = train_test_split(salary,test_size=0.14,random_state=50)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('ingestion of the data has been completed')

            return (self.ingestion_config.train_data_path,
                    
                    self.ingestion_config.test_data_path
                    )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':

    di = DataIngestion()
    salary_train_data,salary_test_data = di.initiate_data_ingestion()

    dt = DataTransformation()
    train_arr,test_arr,preprocessor_path = dt.initiate_datatrans(salary_train_data,salary_test_data)

    mt= ModelTrainer()
    best_model,best_score = mt.initiate_model_training(train_arr,test_arr)

    print(best_model,best_score)




