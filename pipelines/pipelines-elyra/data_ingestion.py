import os
import zipfile
from datetime import datetime
from typing import List

import boto3
import pandas as pd


def decompress_files(zip_filename, output_dir) -> List[str]:
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zipf.extractall(output_dir)
        return [os.path.join(output_dir, name) for name in zipf.namelist()]


def ingest_data(data_folder="/data"):
    print("Commencing data ingestion.")

    s3_endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    s3_endpoint_url = s3_endpoint_url if s3_endpoint_url.startswith('http') else 'http://' + s3_endpoint_url
    s3_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = os.environ.get("AWS_S3_BUCKET")

    print(
        f"Downloading data "
        f'from bucket "{s3_bucket_name}" '
        f"from S3 storage at {s3_endpoint_url}"
    )

    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    # Download the zip file
    downloaded_file = 'data.zip'
    s3_file_path = 'data/data.zip'
    output_dir = 'dataset'
    s3_client.download_file(s3_bucket_name, 'data/data.zip', downloaded_file)

    df = pd.DataFrame(columns=["Date", "Tickets"])
    for f in decompress_files(downloaded_file, output_dir):
        file_name = os.path.basename(f)
        date = datetime.strptime(os.path.splitext(file_name)[0], "%Y%m%d").date()
        tickets_df = pd.read_csv(f)
        n_tickets = len(tickets_df)
        new_row = pd.DataFrame({"Date": [date], "Tickets": [n_tickets]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("data.csv", index=False)

    print("Finished data ingestion.")


if __name__ == "__main__":
    ingest_data(data_folder="data")
