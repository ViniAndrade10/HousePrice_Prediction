import pandas as pd
from pathlib import Path
from controller import ModelCreation


ROOT_PATH = Path(__file__).resolve().parent.parent
dataset_path = ROOT_PATH / "data/housing_dataset.csv"

dataset_houses = pd.read_csv(dataset_path, sep=",")

objetiveModel = ModelCreation(
    dataset_houses, 
    "squared_error", 
    8, 0.1, 200, 
    "median_house_value"
    )

objetiveModel.trainingModel()
