import os
import boto3
import pandas as pd
import re
from datetime import datetime


def ingest_data(data_folder="/data"):
    
    print("Commencing data ingestion.")

    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = os.environ.get("AWS_S3_BUCKET")

    print(
        f"Downloading data"
        f'from bucket "{s3_bucket_name}" '
        f"from S3 storage at {s3_endpoint_url}"
    )

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    # List objects within the specified bucket and folder
    response = s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=data_folder)

    df = pd.DataFrame(columns=["Date", "Tickets"])

    for obj in response.get("Contents", []):
        key = obj["Key"]
        file_name = os.path.basename(key)
        if re.match(r"^\d+\.csv$", file_name):
            s3_client.download_file(s3_bucket_name, key, file_name)
            date = datetime.strptime(os.path.splitext(file_name)[0], "%Y%m%d").date()
            tickets_df = pd.read_csv(file_name)
            n_tickets = len(tickets_df)
            new_row = pd.DataFrame({"Date": [date], "Tickets": [n_tickets]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("data.csv", index=False)

    print("Finished data ingestion.")


if __name__ == "__main__":
    ingest_data(data_folder="/data")
