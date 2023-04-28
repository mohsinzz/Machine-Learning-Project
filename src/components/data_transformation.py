import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransConfig:
    preprocessor_object_file_path :str= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self): 
        self.datatrans_config = DataTransConfig()
    def get_data_tranformation_object(self):
        try:
            numfeatures = ['10percentage','12percentage','collegeGPA',
                           'personality score','aptitude score',
                           'openess_to_experience','Domain',
                           'GraduationYear','Year of Birth']
            
            catfeatures = ['Gender','Degree','CollegeTier',
                           'Specialization','CollegeCityTier']

            # creating numerical and categorical pipelines

            num_pipeline = Pipeline(steps= [('imputer',SimpleImputer(strategy='median')),
                                            ('Scalar',StandardScaler())])
            cat_pipeline = Pipeline(steps= [('imputer',SimpleImputer(strategy='most_frequent')),
                                            ('one-hot-encoder',OneHotEncoder())])

            logging.info('Data Transformation for both ,numerical and categorical features has been done')

            #Combining the num_pipeline and cat_pipeline

            preprocessor = ColumnTransformer([('pipeline_for_num_features',num_pipeline,numfeatures)
                                              ,('pipeline_for_cat_features',cat_pipeline,catfeatures)])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_datatrans(self,train_path,test_path):
        # we are getting train_path and test_path from data ingestion
        try:
            train_salary = pd.read_csv(train_path)
            test_salary = pd.read_csv(test_path)
            preprocessor_object = self.get_data_tranformation_object()

            logging.info('Created the preprocessor Object')

            target= 'Salary'
            salary_train_target = train_salary[[target]]
            salary_test_target = test_salary[[target]]

            salary_num_cat_train = train_salary.drop(columns=[target],axis=1)
            salary_num_cat_test = test_salary.drop(columns=[target],axis=1)
            
            salary_transformed_trained_arr =preprocessor_object.fit_transform(salary_num_cat_train)
            salary_transformed_test_arr =preprocessor_object.transform(salary_num_cat_test)

            salary_train_arr = np.c_[salary_transformed_trained_arr,np.array(salary_train_target)]
            salary_test_arr = np.c_[salary_transformed_test_arr,np.array(salary_test_target)]
            
            logging.info('Created the final train and test arrays')
            # save_object function is created in the utils.py

            save_object(
                file_path = self.datatrans_config.preprocessor_object_file_path,
                obj = preprocessor_object
            )

            logging.info('Saved the preprocessor object in .pkl format')

            return ( salary_train_arr,
                    salary_test_arr,
                    self.datatrans_config.preprocessor_object_file_path
                    )
        
        except Exception as e:
            raise CustomException(e,sys)
            

