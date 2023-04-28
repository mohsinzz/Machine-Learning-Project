import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

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
            salary =pd.read_csv('datafiles\final_salary_prediction_data.csv')
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
    di.initiate_data_ingestion()
