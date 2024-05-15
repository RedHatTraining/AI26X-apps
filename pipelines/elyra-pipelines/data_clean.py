import pandas as pd


def clean_data(data_file="data.csv", data_folder="/data"):
    df = pd.read_cs(f"{data_folder}/{data_file}")
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)
    df["Tickets"].fillna(method="ffill", inplace=True)
    df.to_csv(f"{data_folder}/clean-data.csv")


if __name__ == "__main__":
    clean_data(data_file="data.csv", data_folder="/data")
