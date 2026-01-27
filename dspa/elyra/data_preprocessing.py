import pandas as pd


def clean_data(data_file="data.csv"):
    df = pd.read_csv(data_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(by="Date", inplace=True)
    df["Tickets"].fillna(method="ffill", inplace=True)
    df.to_csv("clean-data.csv", index=False)


if __name__ == "__main__":
    clean_data(data_file="data.csv")
