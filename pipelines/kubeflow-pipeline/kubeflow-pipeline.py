# Copyright 2020 kubeflow.org
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kfp import components
from kfp import dsl


def random_num(low: int, high: int) -> int:
    """Generate a random number between low and high."""
    import random
    return random.randint(low, high)


def flip_coin() -> str:
    """Flip a coin and output heads or tails randomly."""
    import random
    return 'heads' if random.randint(0, 1) == 0 else 'tails'


def print_msg(msg: str):
    """Print a message."""
    print(msg)


# TODO: Create a component from the flip_coin function

# TODO: Create a component from the print_msg function

# TODO: Create a component from the random_num function


# TODO: Define a pipeline
def flipcoin_pipeline():
    flip = flip_coin_op()
    # TODO: Add pipeline logic
