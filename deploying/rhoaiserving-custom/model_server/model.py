import argparse
import os
import pathlib
from typing import Dict, Union

from kserve.errors import InferenceError, ModelMissingError
from kserve.storage import Storage
from kserve.model_server import ModelServer

import joblib
from kserve.protocol.infer_type import InferRequest, InferResponse
from kserve.utils.utils import get_predict_input, get_predict_response
from kserve import Model, model_server

MODEL_EXTENSIONS = (".joblib", ".pkl", ".pickle")


def _find_model_path(model_dir_path):
    for file in os.listdir(model_dir_path):
        file_path = os.path.join(model_dir_path, file)
        if os.path.isfile(file_path) and file.endswith(MODEL_EXTENSIONS):
            return model_dir_path / file
    raise ModelMissingError(model_dir_path)


class SKLearnModel(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_dir = model_dir
        self.ready = False

    def load(self) -> bool:
        model_dir_path = pathlib.Path(Storage.download(self.model_dir))
        model_path = _find_model_path(model_dir_path)
        self._model = joblib.load(model_path)
        self.ready = True
        return self.ready

    def predict(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferResponse]:
        try:
            instances = get_predict_input(payload)
            # result = self._model.predict(instances)
            # return get_predict_response(payload, result, self.name)
            raise InferenceError("Use the model to make predictions to finish the exercise")
        except Exception as e:
            raise InferenceError(str(e))


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_dir", required=True, help="A local path to the model file"
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = SKLearnModel(args.model_name, args.model_dir)
    model.load()
    ModelServer().start([model])
