import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This Funcation is responsible for data transformation

        """
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                           ('scaler',StandardScaler())])
            cat_pipeline = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                           ('one_hot_encoder',OneHotEncoder()),])
            
            logging.info('Numerical columns scaling completed')
            logging.info('categorical Coding encoding completed')

            preprocessor= ColumnTransformer([('num_pipeline',num_pipeline,numerical_columns),
                                              ('cat_pipeline',cat_pipeline,categorical_columns)]
                                              ,remainder='passthrough')
            
            return preprocessor
            
        except Exception as e:
            logging.info('Error')
            raise CustomException(e,sys)# type:ignore
    def initiate_data_trasnsformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read Train and test data completed')
            logging.info('Obtaining Preprocessing object')
            
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'


            input_feature_train_df = train_df.drop(target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                'applying preprocessing object on trainig and testing dataframes'
            )
            
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_array,np.array(target_feature_test_df)]
            logging.info('Saved preprocessing Obj')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            logging.error('Error',exc_info = True)
            raise CustomException(e,sys)# type: ignore
            
