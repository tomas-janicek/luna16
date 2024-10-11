from torch.utils.tensorboard.writer import SummaryWriter

from luna16 import enums, services


def get_tensortboard_writer(
    mode: enums.Mode, registry: services.ServiceContainer
) -> SummaryWriter:
    match mode:
        case enums.Mode.TRAINING:
            return registry.get_service(services.TrainingWriter)
        case enums.Mode.VALIDATING:
            return registry.get_service(services.ValidationWriter)
