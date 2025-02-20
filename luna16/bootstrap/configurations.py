import pydantic

from luna16 import enums

########################
# Model Configurations #
########################


class ModelType(pydantic.BaseModel): ...


class CnnModel(ModelType):
    n_blocks: int


class BiasedModel(CnnModel): ...


class DropoutModel(CnnModel):
    dropout_rate: float


class Dropout3DModel(CnnModel):
    dropout_rate: float


class DropoutOnlyModel(CnnModel):
    dropout_rate: float


class BatchNormalizationModel(CnnModel): ...


class BestCnnModel(DropoutModel): ...


class CnnLoadedModel(CnnModel):
    name: str
    version: str
    finetune: bool
    model_loader: enums.ModelLoader


class BiasedLoadedModel(CnnLoadedModel): ...


class BestCnnLoadedModel(CnnLoadedModel): ...


###########################
# Optimizer Configuration #
###########################


class OptimizerType(pydantic.BaseModel):
    lr: float
    weight_decay: float


class AdamOptimizer(OptimizerType):
    betas: tuple[float, float]


class SgdOptimizer(OptimizerType):
    momentum: float


class BestOptimizer(AdamOptimizer): ...


###########################
# Scheduler Configuration #
###########################


class SchedulerType(pydantic.BaseModel): ...


class StepScheduler(SchedulerType):
    gamma: float


class BestScheduler(StepScheduler): ...
