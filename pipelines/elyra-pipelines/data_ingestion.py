from os import environ
import boto3


def ingest_data(data_folder="./data"):
    print("Commencing data ingestion.")

    s3_endpoint_url = environ.get("AWS_S3_ENDPOINT")
    s3_access_key = environ.get("AWS_ACCESS_KEY_ID")
    s3_secret_key = environ.get("AWS_SECRET_ACCESS_KEY")
    s3_bucket_name = environ.get("AWS_S3_BUCKET")

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
    files = [obj["Key"] for obj in response["Contents"]]

    for file in files:
        s3_client.download_file(s3_bucket_name, file, f"{data_folder}/{file}")

    print("Finished data ingestion.")


if __name__ == "__main__":
    ingest_data(data_folder="/data")
