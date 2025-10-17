import sys
import requests
from urllib3.exceptions import InsecureRequestWarning
from transformers import DistilBertTokenizer

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

GREEN = "\033[92m"
RESET = "\033[0m"
BOLD = "\033[1m"


def tokenize(text: str):
    tokens = tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )

    # Extract input IDs and attention mask
    input_ids = tokens["input_ids"].tolist()[0]
    attention_mask = tokens["attention_mask"].tolist()[0]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def prepare_distilbert_request(tokens):
    return {
        "instances": [
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }
        ]
    }


def prepare_iris_request():
    return {
        "inputs": [
            {"name": "X", "shape": [1, 4], "datatype": "FP32", "data": [3, 4, 3, 2]}
        ]
    }


def prepare_diabetes_request():
    return {
        "inputs": [
            {"name": "dense_input", "shape": [1, 8], "datatype": "FP32", "data": [6.0, 110.0, 65.0, 15.0, 1.0, 45.7, 0.627, 50.0]}
        ]
    }


def send_inference_request(url, body, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return requests.post(url, json=body, headers=headers, verify=False)


def print_curl_request(url, query):
    print(
        f'\n{BOLD}{GREEN}Inferece request for the {url} url, using "{query}" as input.{RESET}\n'
    )
    # Tokenize the input text
    tokens = tokenize(query)

    # Define request and print
    body = f"""'{{"instances": [
        {{
            "input_ids": [{", ".join([str(i) for i in tokens["input_ids"]])}],
            "attention_mask": [{", ".join([str(i) for i in tokens["attention_mask"]])}]
        }}
    ]}}'
    """
    request = f'curl -X POST -k {url} \\ \n -H "Content-Type: application/json" \\ \n -d {body}'
    print(request)


if __name__ == "__main__":
    query = sys.argv[2]
    url = sys.argv[1]
    print_curl_request(url, query)
