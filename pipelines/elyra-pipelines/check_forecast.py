import boto3
import pandas as pd
import matplotlib.pyplot as plt


def check_forecast(data_folder="/data"):

    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = os.environ.get("AWS_S3_BUCKET")

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    s3_client.download_file(s3_bucket_name, "/data/clean_data.csv", "./clean_data.csv")
    s3_client.download_file(
        s3_bucket_name, "/data/forecast_data.csv", "./forecast_data.csv"
    )

    clean_data = pd.read_csv("./clean_data.csv")
    forecast_data = pd.read_csv("./clean_data.csv")

    clean_data["Date"] = pd.to_datetime(clean_data["Date"])
    forecast_data["Date"] = pd.to_datetime(forecast_data["Date"])

    # Plot the data
    plt.figure(figsize=(12, 6))

    plt.plot(clean_data["Date"], clean_data[clean_data.columns[1]], label="data")
    plt.plot(
        forecast_data["Date"], forecast_data[forecast_data.columns[1]], label="forecast"
    )

    plt.xlabel("Date")
    plt.ylabel("# of tickets")
    plt.legend()
    plt.grid(True)
    plt.show()
