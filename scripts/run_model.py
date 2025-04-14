import csv
import sys
from pathlib import Path

import joblib
import pandas as pd
from data_loader_and_featurizer import data_path, run_featurizer


def save_smiles_to_csv(smiles_string, filename="smiles.csv"):
    file_path = data_path / filename
    with file_path.open(mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerow({"smiles": smiles_string})


# Example usage
user_input = input("Enter a SMILES string (or type 'quit'/'q' to exit): ")
if user_input.strip().lower() in ("quit", "q"):
    print("Exiting the program...")
    sys.exit()  # Completly exiting the program.
else:
    save_smiles_to_csv(user_input)


morgan_id = "eos4wt0"

comp_id = "eos2gw4"
# Run the Morgan featurizer
run_featurizer(
    input_filenames=["smiles.csv"],
    output_filenames=["smiles_morgan.csv"],
    model_id=morgan_id,
    data_path=data_path,
)
# Run the Comp featurizer
run_featurizer(
    input_filenames=["smiles.csv"],
    output_filenames=["smiles_comp.csv"],
    model_id=comp_id,
    data_path=data_path,
)

# Read the output files
comp_df = pd.read_csv(data_path / "smiles_comp.csv").drop(columns=["key", "input"])
morgan_df = pd.read_csv(data_path / "smiles_morgan.csv").drop(columns=["key", "input"])
data_df = pd.concat([comp_df, morgan_df], axis=1)

model_path = Path(__file__).parent.parent / "models"
# Load the model
rf_model = joblib.load(model_path / "rf_model.joblib")
# Replace your DataFrame's columns with the expected ones
expected_columns = rf_model.feature_names_in_
data_df.columns = expected_columns

predictions = rf_model.predict(data_df)


# Get the first prediction
prediction = predictions[0]

if prediction == 1:
    print(f"🔴 The molecule:{user_input} is predicted to be a **skin sensitizer**.")
else:
    print(f"🟢 The molecule:{user_input} is predicted to **not be a skin sensitizer**.")
