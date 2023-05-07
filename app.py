from flask import Flask,render_template,request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import predict_pipeline
application = Flask(__name__)
app=application

@app.route('/',methods=['GET','POST'])
def index():
    if request.method =='GET':
        return render_template('home.html')
    else:
        return render_template('salary.html')

@app.route('/details',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('salary.html')
    else:
        data =   CustomData(tenth_percent =float(request.form.get('10thpercent')),
                          
                            twelfth_percent=float(request.form.get('12thpercent')),

                            CollegeGPA =float(request.form.get('Gradpercent')), 

                            personality_score=float((float(request.form.get('Conscientiousness'))+float(request.form.get('Agreeableness'))
                                                  +float(request.form.get('Extraversion'))+float(10)-float(request.form.get('Neuroticism')))/4),

                            aptitude_score=float((float(request.form.get('Quants'))+float(request.form.get('Verbal'))
                                                  +float(request.form.get('Logical')))/3),
                            openness_to_experience=float(request.form.get('openness_to_experience')),

                            Domain=float(request.form.get('Domain')),

                            GraduationYear = float(request.form.get('GraduationYear')),

                            Year_of_Birth=float(request.form.get('Year_of_birth')),

                            Gender=str(request.form.get('Gender')).lower(),

                            Degree=str(request.form.get('Degree')),

                            CollegeTier= str('Tier')+str(request.form.get('CollegeTier'))+str('College'),

                            Specialization=str(request.form.get('Specialization')),

                            CollegeCityTier=str('CityTier') +str(request.form.get('CollegeCityTier')))
        
        salary_pred_input_df= data.get_dataframe()
        predicted_salary = predict_pipeline().prediction(salary_pred_input_df)
        return render_template('salary.html',results=predicted_salary)
    
if __name__=='__main__':
    app.run(host='0.0.0.0')

    

