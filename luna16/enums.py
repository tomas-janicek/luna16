import enum


class DimensionIRC(enum.IntEnum):
    INDEX = 0
    ROW = 1
    COL = 2


class Mode(enum.Enum):
    TRAINING = "Training"
    VALIDATING = "Validating"


class UpMode(enum.Enum):
    UP_CONV = enum.auto
    UP_SAMPLE = enum.auto


class LunaCandidateTypes(enum.IntEnum):
    POSITIVE = 0
    NEGATIVE = 1


class CandidateClass(enum.IntEnum):
    MALIGNANT = 0
    BENIGN = 1
    NOT_NODULE = 2


class ModelLoader(enum.Enum):
    ML_FLOW = "mlflow"
    FILE = "file"
