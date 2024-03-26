from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_training_data(
        csv_path: Path, test_size: float = 0.2, random_state: int = 0
) -> List:
    path = Path(csv_path)
    data = pd.read_csv(path.as_posix())

    x = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)
