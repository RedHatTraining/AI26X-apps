from kfp import compiler, kubernetes
from kfp.dsl import (
    component,
    pipeline,
    Input,
    Output,
    Dataset,
    importer,
    Model,
    Metrics,
    ClassificationMetrics,
    Markdown,
)

DATA_SCIENCE_IMAGE = "quay.io/modh/runtime-images:runtime-datascience-ubi9-python-3.9-2024a-20241011"  # noqa


@component(base_image=DATA_SCIENCE_IMAGE, packages_to_install=["psycopg2==2.9.10"])
def query_db_data(db_host: str, dataset: Output[Dataset]):
    import os
    import psycopg2

    # Get the DB connection parameters injected as environment variables
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    # Connect to the database
    with psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
    ) as connection:
        with connection.cursor() as cursor:
            # Query the data
            cursor.execute("SELECT * FROM Sentiment")
            rows = cursor.fetchall()

    # Write the data in CSV format to the output path
    with open(dataset.path, "w") as f:
        f.write("comment,sentiment\n")
        for comment, sentiment in rows:
            f.write(f'"{comment}",{sentiment}\n')


@component(base_image=DATA_SCIENCE_IMAGE)
def integrate_and_preprocess_data(
    s3_dataset: Input[Dataset],
    db_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Read data from the two datasets
    data_from_s3 = pd.read_csv(s3_dataset.path)
    data_from_db = pd.read_csv(db_dataset.path)

    # Combine rows into a single dataframe
    dataset = pd.concat([data_from_s3, data_from_db], axis=0)

    # Convert sentiment text into numeric labels
    class_ids = {"positive": 1, "negative": 0}
    dataset["label"] = dataset.sentiment.map(class_ids)

    # Generate train / test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.comment, dataset.label, test_size=0.2, random_state=83
    )

    # Save train / test datasets
    train_data = pd.DataFrame({"text": X_train, "label": y_train})
    train_data.to_csv(train_dataset.path, index=False)

    test_data = pd.DataFrame({"text": X_test, "label": y_test})
    test_data.to_csv(test_dataset.path, index=False)


@component(base_image=DATA_SCIENCE_IMAGE)
def train_model(dataset: Input[Dataset], model: Output[Model]):
    import joblib
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    train_dataset = pd.read_csv(dataset.path)
    X_train = train_dataset.text
    y_train = train_dataset.label

    # Create the model
    classifier = make_pipeline(
        TfidfVectorizer(), MultinomialNB()  # Converts text into TF-IDF features
    )

    # Train the model
    classifier.fit(X_train, y_train)

    # Save the model
    joblib.dump(classifier, model.path)


@component(base_image=DATA_SCIENCE_IMAGE)
def evaluate_model(
    model: Input[Model],
    test_dataset: Input[Dataset],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    report: Output[Markdown],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    # Load the model
    model = joblib.load(model.path)

    # Load the test set
    train_dataset = pd.read_csv(test_dataset.path)
    X_test = train_dataset.text
    y_test = train_dataset.label

    # Predict the sentiment on the test set
    predictions = model.predict(X_test)

    # Compute accuracy comparing the expected results (y_test)
    # and the predictions
    score = accuracy_score(y_test, predictions)

    classes = ["negative", "positive"]
    conf_matrix = confusion_matrix(y_test, predictions).tolist()
    # Log metrics
    metrics.log_metric("accuracy", score)
    classification_metrics.log_confusion_matrix(classes, conf_matrix)

    # Log a markdown report
    content = "# Sentiment Prediction Report on Test Set\n\n"
    for text, prediction, expected in zip(X_test, predictions, y_test):
        predicted_class = classes[int(prediction)]
        expected_class = classes[int(expected)]
        content += f"- Text: {text}, Prediction: {predicted_class}, Expected: {expected_class}\n"

    # Write the report
    with open(report.path, "w") as f:
        f.write(content)


@pipeline
def pipeline(
    db_host: str = "postgres",
    s3_data_path: str = "s3://manage-artifacts/data/data_s3.csv",
) -> Model:

    # Get data from DB
    read_db_data_task = query_db_data(db_host=db_host)
    read_db_data_task.set_caching_options(False)
    # Inject database connection paramaters as environment variables
    kubernetes.use_secret_as_env(
        read_db_data_task,
        secret_name="postgres",
        secret_key_to_env={"database-name": "DB_NAME"},
    )
    kubernetes.use_secret_as_env(
        read_db_data_task,
        secret_name="postgres",
        secret_key_to_env={"database-user": "DB_USER"},
    )
    kubernetes.use_secret_as_env(
        read_db_data_task,
        secret_name="postgres",
        secret_key_to_env={"database-password": "DB_PASSWORD"},
    )

    # Get data from S3
    read_s3_data_task = importer(
        artifact_uri=s3_data_path,
        artifact_class=Dataset,
        reimport=False,
    )

    # Integrate both datasets into one training dataset,
    # preprocess and clean data
    preprocess_task = integrate_and_preprocess_data(
        s3_dataset=read_s3_data_task.output, db_dataset=read_db_data_task.output
    )

    # Train the model
    train_task = train_model(dataset=preprocess_task.outputs["train_dataset"])
    model = train_task.output

    # Evaluate the model
    evaluate_model(model=model, test_dataset=preprocess_task.outputs["test_dataset"])

    return model


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipeline.yaml")
