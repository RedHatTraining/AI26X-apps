import os
import pandas as pd
from datetime import datetime


def preprocess(data_folder="./data"):
    print("preprocessing data")
    files = os.listdir(data_folder)
    df = pd.DataFrame(columns=["Date", "Tickets"])
    for file in files:
        date = datetime.strptime(os.path.splitext(file)[0], "%Y%m%d").date()
        tickets_df = pd.read_csv(f"{data_folder}/{file}")
        n_tickets = len(tickets_df)
        df = df.append({"Date": date, "Tickets": n_tickets}, ignore_index=True)
    df.to_csv(f"{data_folder}/data.csv")


if __name__ == "__main__":
    preprocess(data_folder="/data")
