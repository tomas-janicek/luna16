from luna16 import enums, services

from . import model_factories, service_factories


def create_registry(
    model_type: enums.ModelType,
    optimizer_type: enums.OptimizerType = enums.OptimizerType.ADAM,
    scheduler_type: enums.SchedulerType = enums.SchedulerType.STEP,
) -> services.ServiceContainer:
    registry = services.ServiceContainer()
    service_factory = service_factories.ServiceFactory(registry)
    model_factory = model_factories.ModelFactory(registry)

    service_factory.add_tensorboard_writers()
    service_factory.add_mlflow_run()
    service_factory.add_model_savers()
    service_factory.add_hyperparameters()
    service_factory.add_message_handler()

    model_factory.add_model(model_type)
    model_factory.add_optimizer(optimizer_type)
    model_factory.add_scheduler(scheduler_type)

    return registry
