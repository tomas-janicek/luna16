import typing
from dataclasses import dataclass

import numpy as np


@dataclass
class Value:
    name: str
    value: typing.Any


@dataclass
class NumberValue(Value):
    name: str
    value: float | np.float_
    formatted_value: str
