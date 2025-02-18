from luna16 import services

from . import configurations, model_factories, service_factories


def create_registry(
    model_type: configurations.ModelType,
    optimizer_type: configurations.OptimizerType,
    scheduler_type: configurations.SchedulerType,
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


def create_tunning_registry(
    model_type: configurations.ModelType,
    optimizer_type: configurations.OptimizerType,
    scheduler_type: configurations.SchedulerType,
) -> services.ServiceContainer:
    registry = services.ServiceContainer()
    service_factory = service_factories.ServiceFactory(registry)
    model_factory = model_factories.ModelFactory(registry)

    service_factory.add_hyperparameters()
    service_factory.add_message_handler()

    model_factory.add_model(model_type)
    model_factory.add_optimizer(optimizer_type)
    model_factory.add_scheduler(scheduler_type)

    return registry
