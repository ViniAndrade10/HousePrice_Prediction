# type: ignore
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
import os
# Machine Learning Processing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from data_treatment import TreatmentData

ROOT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_PATH / "model"

os.system("cls")
warnings.filterwarnings("ignore")

class ModelCreation:

    def __init__(self, dataset:pd.DataFrame, loss, max_depth, learning_rate, n_estimators, col_objective) -> None:
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.dataset = dataset
        self.loss = loss
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.col_objective = col_objective

        self.separatingData()

    def separatingData(self):
        self.dataset = self.dataset.join(pd.get_dummies(self.dataset["ocean_proximity"], dtype=int), how="inner")
        self.dataset["room_index"] = self.dataset["total_rooms"] / self.dataset["total_bedrooms"]
        self.dataset.dropna(inplace=True)

        y = self.dataset[self.col_objective]
        X = self.dataset
        X.drop([y.name, "ocean_proximity"], axis=1, inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.3,
            shuffle=True
            )

    def trainingModel(self):

        model = GradientBoostingRegressor(
            loss=self.loss,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators
            )

        # Training Model
        model.fit(self.X_train, self.y_train)
        resultModel = model.predict(self.X_test)

        r2_Model = r2_score(self.y_test, resultModel)
        print(f"Model's r² Score: {r2_Model * 100}")

        if r2_Model <= 0.6:
            print("Model with low performance. Evaluate new training")
        else:
            pickle.dump(model, open(MODEL_PATH / self.filename, "wb"))
            print("Model Trainned Successfully!")

class ModelPrediction:
    def __init__(self) -> None:
        self.model = None
        self.filename = "housePredictionModel.sav"


        self.readModel()

    def readModel(self):
        self.model = pickle.load(open(MODEL_PATH / self.filename, "rb"))

    def makingPredictions(self, dataset_input:pd.DataFrame):
        dataForPrediction = TreatmentData(dataset_input)
        prediction = self.model.predict(dataForPrediction.dataset_input)
        datasetOutput = dataForPrediction.dataset_input
        datasetOutput["Prediction"] = prediction
        return datasetOutput
