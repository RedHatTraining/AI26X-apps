import sys

from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'


def tokenize(text: str):
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    # Extract input IDs and attention mask
    input_ids = tokens["input_ids"].tolist()[0]
    attention_mask = tokens["attention_mask"].tolist()[0]
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def gen_request(url, query):
    # Input text for sentiment analysis
    text = "This model is great!"

    print(f'\n{BOLD}{GREEN}Inferece request for the {url} url, using "{query}" as input.{RESET}\n')
    # Tokenize the input text
    tokens = tokenize(text)

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


if __name__ == '__main__':
    query = sys.argv[2]
    url = sys.argv[1]
    gen_request(url, query)
