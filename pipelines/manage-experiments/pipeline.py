from typing import Literal
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


@component(base_image=DATA_SCIENCE_IMAGE)
def preprocess_data(
    s3_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Read data from CSV file
    dataset = pd.read_csv(s3_dataset.path)

    # Convert sentiment text into numeric labels
    class_ids = {"positive": 1, "negative": 0}
    dataset["label"] = dataset.sentiment.map(class_ids)

    # Generate train / test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.comment, dataset.label, test_size=0.2, random_state=15
    )

    # Save train / test datasets
    train_data = pd.DataFrame({"text": X_train, "label": y_train})
    train_data.to_csv(train_dataset.path, index=False)

    test_data = pd.DataFrame({"text": X_test, "label": y_test})
    test_data.to_csv(test_dataset.path, index=False)


@component(base_image=DATA_SCIENCE_IMAGE)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model],
    classifier_name: str,
):
    import joblib
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC

    data = pd.read_csv(dataset.path)
    X_train = data.text
    y_train = data.label

    # Available methods to implement a classifier
    methods = {
        "MultinomialNB": lambda: MultinomialNB(),
        "LogisticRegression": lambda: LogisticRegression(),
        "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=0),
        "SVC": lambda: SVC(kernel="linear"),
    }

    # Select the classifier to use
    if classifier_name not in methods:
        raise RuntimeError(
            f"{classifier_name} is not a valid classifier. "
            "Choose one of: " + (", ".join(methods.keys()))
        )

    method = methods[classifier_name]

    # Create the model
    classifier = make_pipeline(TfidfVectorizer(), method())

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
    data = pd.read_csv(test_dataset.path)
    X_test = data.text
    y_test = data.label

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
    classifier_name: str = "MultinomialNB",
    s3_data_path: str = "s3://manage-experiments/data/data_s3.csv",
) -> Model:

    # Get data from S3
    read_s3_data_task = importer(
        artifact_uri=s3_data_path,
        artifact_class=Dataset,
        reimport=False,
        metadata={"name": "raw_data"}
    )

    # Integrate both datasets into one training dataset,
    # preprocess and clean data
    preprocess_task = preprocess_data(s3_dataset=read_s3_data_task.output)

    # Train the model
    train_task = train_model(
        classifier_name=classifier_name,
        dataset=preprocess_task.outputs["train_dataset"],
    )
    model = train_task.output

    # Evaluate the model
    evaluate_model(model=model, test_dataset=preprocess_task.outputs["test_dataset"])

    return model


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipeline.yaml")
