from typing import List, NamedTuple
from kfp import dsl, compiler


PYTHON_IMAGE = "registry.access.redhat.com/ubi9/python-39:1-197.1726696853"  # noqa
DATA_SCIENCE_IMAGE = "quay.io/modh/runtime-images:runtime-datascience-ubi9-python-3.9-2024a-20241011"  # noqa


# Component to process the dataset
@dsl.component(base_image=DATA_SCIENCE_IMAGE)
def process_data() -> NamedTuple("outputs", texts=List[str], labels=List[int]):
    # Sample dataset
    dataset = [
        ("I love this!", "positive"),
        ("I hate this!", "negative"),
        ("This is awesome! ", "positive"),
        ("This is terrible!", "negative"),
        ("I really enjoyed the experience.", "positive"),
        ("I did not like the experience at all.  ", "negative"),
        ("  I love it.", "positive"),
        ("I hate it.", "negative"),
        ("I like it.", "positive"),
        ("I don't like it.", "negative"),
        ("I like this.", "positive"),
        ("   That is not good.", "negative"),
        ("That is so cool, love it.", "positive"),
        ("I had a bad experience.", "negative"),
        ("That is awesome", "positive"),
        ("   I am worried about it.   ", "negative"),
        ("This is wonderful", "positive"),
        ("This is terrible, don't even try", "negative"),
        ("This is amazing!", "positive"),
        ("That is not my cup of tea.", "negative"),
        ("I like it so much   .", "positive"),
    ]

    # Separate texts and labels into different lists
    texts = [sample[0].strip() for sample in dataset]
    labels = [sample[1] for sample in dataset]

    # Convert labels into numbers
    class_ids = {"positive": 1, "negative": 0}
    labels = [class_ids[label] for label in labels]

    # Return texts and labels
    outputs = NamedTuple("outputs", texts=List[str], labels=List[int])
    return outputs(texts, labels)


# Component to train the model
@dsl.component(base_image=DATA_SCIENCE_IMAGE)
def train_model(texts: list, labels: list) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split dataset into training and test sets (even though it's tiny)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=83
    )

    # Create a pipeline with a TF-IDF Vectorizer and a Logistic Regression classifier
    pipeline = make_pipeline(
        TfidfVectorizer(), MultinomialNB()  # Converts text into TF-IDF features
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)

    print("Test texts:", X_test)
    print("Test predictions:", predictions)
    print("Expected prediction:", y_test)

    return accuracy


# Component to verify the accuracy of the model
@dsl.component(base_image=PYTHON_IMAGE)
def verify_accuracy(accuracy: float, threshold: float):
    import sys

    if accuracy >= threshold:
        print("Model trained successfully")
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"The model did not achieve the minimum accuracy of {threshold * 100:.2f}%.")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        sys.exit(1)


# The pipeline
@dsl.pipeline(name="sentiment-analysis")
def pipeline():
    # Load and preprocess data
    data_processing_task = process_data()
    texts = data_processing_task.outputs["texts"]
    labels = data_processing_task.outputs["labels"]

    # Train the model
    train_task = train_model(texts=texts, labels=labels)
    accuracy = train_task.output

    # Verify the model accuracy
    verify_accuracy(accuracy=accuracy, threshold=0.5)


if __name__ == "__main__":
    outfile = "pipeline.yaml"
    # Compile the pipeline
    compiler.Compiler().compile(pipeline, outfile)
    print(
        "Pipeline compiled.\n"
        f"Use the RHOAI dashboard to import the '{outfile}' file"
    )
