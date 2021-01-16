import os
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import numpy as np

# Set parameters

TRAIN_PATH = "data/red_wine_train.csv"
VALID_PATH = "data/red_wine_valid.csv"
TEST_PATH = "data/red_wine_test.csv"

VALID_FRACTION = 0.1
TEST_FRACTION = 0.2

SEED = 5

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def main():

    # Check if the files to be created already exist
    paths = [TRAIN_PATH, VALID_PATH, TEST_PATH]

    if sum([Path(x).is_file() for x in paths]) == len(paths):
        return "Data files already prepared"

    else:
        # Create data folder if required
        if not os.path.exists("./data"):
            os.makedirs("./data")

        # Download the data from the repository
        print("Downloading data...")
        df_all = pd.read_csv(URL, sep=";")

        # Split the data
        print("Splitting data...")
        df_train_valid, df_test = train_test_split(
            df_all, test_size=TEST_FRACTION, random_state=SEED
        )
        df_train, df_valid = train_test_split(
            df_train_valid,
            test_size=VALID_FRACTION / (1 - TEST_FRACTION),
            random_state=SEED,
        )

        # Check the splits are as specifiec
        assert np.isclose(
            round(df_train.shape[0] / df_all.shape[0], 2),
            1 - TEST_FRACTION - VALID_FRACTION,
        ), "Training set incorrect size"
        assert np.isclose(
            round(df_valid.shape[0] / df_all.shape[0], 2), VALID_FRACTION
        ), "Validation size incorrect size"
        assert np.isclose(
            round(df_test.shape[0] / df_all.shape[0], 2), TEST_FRACTION
        ), "Test set incorrect size"

        # Save the splits
        df_train.to_csv(TRAIN_PATH)
        df_valid.to_csv(VALID_PATH)
        df_test.to_csv(TEST_PATH)

        # Confirm all the expected files now exist
        if sum([Path(x).is_file() for x in paths]) == len(paths):
            print("Data downloaded and split")
        else:
            print("Download and split may not have completed correctly")


if __name__ == "__main__":
    main()
