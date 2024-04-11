import os
import pandas as pd
from sklearn.datasets import make_classification


def generate_synthetic_dataset(
    num_features: int,
    num_samples: int,
    file_path: str,
    num_informative_features: int,
    num_classes=2,
    chunk_size=1000,
    random_state=42,
):
    chunk_size = min(chunk_size, num_samples)
    num_chunks = num_samples // chunk_size

    # create CSV file and write header
    header = [f"feature_{i+1}" for i in range(num_features)] + ["target"]
    with open(file_path, "w") as f:
        f.write(",".join(header) + "\n")

    for i in range(num_chunks):
        # generate  data with distinct clusters for each class
        X, y = make_classification(
            n_samples=chunk_size,
            n_features=num_features,
            n_informative=num_informative_features,
            n_classes=num_classes,
            random_state=random_state,
        )

        # create and append data frame to file
        df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(num_features)])
        df["target"] = y

        df.to_csv(file_path, mode="a", header=False, index=False)

        # print percentage complete
        print(f"\r\033[K{100 * (i+1) // num_chunks}%", end="")

    print(f"\nCSV file '{file_path}' generated")
