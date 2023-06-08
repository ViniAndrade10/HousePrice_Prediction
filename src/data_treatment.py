import pandas as pd

class TreatmentData:
    def __init__(self, dataset_input:pd.DataFrame) -> None:
        self.dataset_input = dataset_input

        self.treatingInputData()

    def treatingInputData(self):
        for row in self.dataset_input.index:
            ocean_proximity = self.dataset_input.loc[row, "ocean_proximity"]

            self.dataset_input["<1H OCEAN"] = 0
            self.dataset_input["INLAND"] = 0
            self.dataset_input["ISLAND"] = 0
            self.dataset_input["NEAR BAY"] = 0
            self.dataset_input["NEAR OCEAN"] = 0
            # Changing binary value
            self.dataset_input[ocean_proximity] = 1

            self.dataset_input["room_index"] = self.dataset_input["total_rooms"] / self.dataset_input["total_bedrooms"]

        self.dataset_input.drop(["ocean_proximity"], axis=1, inplace=True)


