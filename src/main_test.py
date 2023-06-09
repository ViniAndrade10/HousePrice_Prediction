import pandas as pd
from pathlib import Path
import os
import warnings
from controller import ModelPrediction


os.system("cls")
warnings.filterwarnings("ignore")

ROOT_PATH = Path(__file__).resolve().parent.parent
datasetInputPath = ROOT_PATH / "data/InputDataFirstPrediction.xlsx"
datasetOutputPath = ROOT_PATH / "export/OutputDataPrediction.xlsx"

datasetInput = pd.read_excel(datasetInputPath)

modelPredictor = ModelPrediction()

predictions = modelPredictor.makingPredictions(datasetInput)

print(predictions)