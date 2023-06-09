import pandas as pd
from pathlib import Path
from io import BytesIO
import os
import warnings
import streamlit as st
from controller import ModelPrediction


os.system("cls")
warnings.filterwarnings("ignore")

# ROOT_PATH = Path(__file__).resolve().parent.parent
# datasetInputPath = ROOT_PATH / "data/InputDataFirstPrediction.xlsx"
# datasetOutputPath = ROOT_PATH / "export/OutputDataPrediction.xlsx"

c1 = st.container()
c1.title("California House Price Prediction")
c1.header(
    "Model created to predict most suitable prices for Houses in California"
    )

c2 = st.container()
c2.subheader("Prediction Area:")
data = c2.file_uploader("Upload data for prediction: ")
button_input = c2.button("Start Prediction")

if button_input:
    datasetInput = pd.read_excel(data)
    modelPredictor = ModelPrediction()
    predictions = modelPredictor.makingPredictions(datasetInput)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        predictions.to_excel(excel_writer=writer,sheet_name="Predictions", index=False)

    c2.download_button(
        "Download Prediction", 
        data=buffer, 
        file_name="HousePrice_Predictions.xlsx"
        )








