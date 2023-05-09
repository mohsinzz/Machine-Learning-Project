
import sys
from src.logger import logging
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class predict_pipeline:
    def __init__(self):
        pass
    def prediction(self,features):
        try:
            modeltrainer_path = 'artifacts/modeltrainer.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = modeltrainer_path)
            preprocessor = load_object(file_path=preprocessor_path)

            transformed_data = preprocessor.transform(features)
            predicted_sal= model.predict(transformed_data)

            return predicted_sal

        except Exception as e:
            raise CustomException(e,sys)




class CustomData:
    # The arguments in init will be coming from the web app
    def __init__(self,tenth_percent : float,twelfth_percent: float,CollegeGPA:float,personality_score:float,
                 aptitude_score:float,openness_to_experience:float,Domain:float,
                 GraduationYear:float,Year_of_Birth:float,Gender:str,Degree:str,
                 CollegeTier:str,
                Specialization:str,CollegeCityTier:str):
        
        self.tenth_percent = tenth_percent

        self.twelfth_percent=twelfth_percent

        self.CollegeGPA = CollegeGPA

        self.personality_score = personality_score

        self.aptitude_score = aptitude_score

        self.openness_to_experience = openness_to_experience

        self.Domain = Domain

        self.GraduationYear = GraduationYear

        self.Year_of_Birth = Year_of_Birth

        self.Gender = Gender

        self.Degree = Degree

        self.CollegeTier = CollegeTier

        self.Specialization = Specialization

        self.CollegeCityTier = CollegeCityTier

    def get_dataframe(self):
        try:
            salary_data_dict = {'10percentage':[self.tenth_percent],'12percentage':[self.twelfth_percent],
                                'collegeGPA':[self.CollegeGPA],
                                'personality score':[self.personality_score],'aptitude score':[self.aptitude_score],
                                'openess_to_experience':[self.openness_to_experience],'Domain':[self.Domain],
                                'GraduationYear':[self.GraduationYear],'Year of Birth':[self.Year_of_Birth],
                                'Gender':[self.Gender],'Degree':[self.Degree],'CollegeTier':[self.CollegeTier],
                                'Specialization':[self.Specialization],'CollegeCityTier':[self.CollegeCityTier]}
            
            return pd.DataFrame(salary_data_dict)
        except Exception as e:
            raise CustomException(e,sys)
        


            




