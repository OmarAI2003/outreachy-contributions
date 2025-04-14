# Import necessary libraries.
from pathlib import Path
from typing import List

from ersilia import ErsiliaModel  # type: ignore

# from tdc.single_pred import Tox # type: ignore
# import pandas as pd

# Set the path to the data directory.
data_path = Path(__file__).parent.parent / "data"


def main():
    pass
    # # Obtain Tox dataset for "Skin Reaction".
    # data = Tox(
    #     name="Skin Reaction",
    #     path=data_path,
    #     print_stats=False,
    # )
    # # Split the dataset with a fixed seed for reproducibility
    # split = data.get_split(method="scaffold", seed=42, frac=[0.8, 0.1, 0.1])
    # train_df, valid_df, test_df = (
    #     pd.DataFrame(split["train"]),
    #     pd.DataFrame(split["valid"]),
    #     pd.DataFrame(split["test"]),
    # )

    # datasets = {"train": train_df, "valid": valid_df, "test": test_df}

    # for name, df in datasets.items():
    #     df[["Drug_ID", "Drug"]].rename(columns={"Drug": "smiles"}).to_csv(
    #         data_path / f"{name}_features.csv", index=False
    #     )
    #     df[["Y"]].to_csv(data_path / f"{name}_labels.csv", index=False)

    # df = pd.read_csv(data_path / "skin_reaction.tab", delimiter="\t")
    # df.to_csv(data_path / "skin_reaction.csv", index=False)


# Delete unnecessary files after splitting data.
# for file in ["skin_reaction.tab"]:
#     os.remove(os.path.join(data_path, file))


def run_featurizer(
    data_path: Path,
    model_id: str,
    input_filenames: List[str],
    output_filenames: List[str],
) -> None:
    """
    Processes multiple input files with a specified model and saves the results to output files.

    Parameters
    ----------
    data_path : Path
        The directory path containing the input and output files.
    model_id : str
        The ID of the model to be used for processing.
    input_filenames : list of str
        A list of input filenames to be processed.
    output_filenames : list of str
        A list of output filenames where results will be saved.

    Raises
    ------
    ValueError
        If the number of input files does not match the number of output files.

    Notes
    -----
    The function processes each input file one by one and saves the results
    to the corresponding output file. The model must be closed after processing all files.
    """
    model = ErsiliaModel(model=model_id)
    model.serve()

    # Ensure input_filenames and output_filenames have the same length
    if len(input_filenames) != len(output_filenames):
        raise ValueError(
            "The number of input files must match the number of output files."
        )

    for input_filename, output_filename in zip(input_filenames, output_filenames):
        input_file = str(data_path / input_filename)
        output_file = str(data_path / output_filename)

        model.run(input=input_file, output=output_file)

    model.close()


if __name__ == "__main__":
    main()
