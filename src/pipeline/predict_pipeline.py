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

