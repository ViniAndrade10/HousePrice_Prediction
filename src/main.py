import pandas as pd
from pathlib import Path
from controller import ModelPrediction
from data_treatment import TreatmentData

ROOT_PATH = Path(__file__).resolve().parent.parent
datasetInputPath = ROOT_PATH / "data/InputDataFirstPrediction.xlsx"
datasetOutputPath = ROOT_PATH / "export/OutputDataPrediction.xlsx"

datasetInput = pd.read_excel(datasetInputPath)

modelPredictor = ModelPrediction()

predictions = modelPredictor.makingPredictions(datasetInput)

with pd.ExcelWriter(datasetOutputPath) as writer:
    predictions.to_excel(
        excel_writer=writer,
        sheet_name="Predictions",
        index=False
    )