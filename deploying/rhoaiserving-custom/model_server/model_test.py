import requests
from kserve import InferRequest, InferInput


def test_model_joblib():
        # test v2 infer call
    infer_input = InferInput(
        name="input-0", shape=[1, 4], datatype="FP32", data=request
    )
    infer_request = InferRequest(model_name="model", infer_inputs=[infer_input])
    infer_response = requests.post("http://localhost:8080")

    assert infer_response.to_rest()["outputs"][0]["data"] == [0]