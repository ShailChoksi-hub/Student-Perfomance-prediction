import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl') #in the source code the name of the pickel file is proprocessor.pkl


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            '''this block is responsible for data transformation'''
            num_columns=["reading_score","writing_score"]
            cat_columns=[ "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course",]

            num_pipeline=Pipeline(steps=[("Standard_scaler",StandardScaler()),("Stndard_Imputer",SimpleImputer(strategy="median"))])
            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("onehotencoder",OneHotEncoder()),("Standard_scaler",StandardScaler(with_mean=False))])

            preprocessor=ColumnTransformer(transformers=[('num',num_pipeline,num_columns),('categorical',cat_pipeline,cat_columns)])
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("datasets have been loaded")
            
            target_column_name="math_score"
            preprocessing_obj= self.get_data_transformation_object()

            input_feature_train_df=train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)