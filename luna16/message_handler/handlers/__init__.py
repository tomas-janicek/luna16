from .. import messages
from .handlers import (
    log_batch_end_to_console,
    log_batch_start_to_console,
    log_epoch_to_console,
    log_metrics_to_console,
    log_metrics_to_mlflow,
    log_metrics_to_tensorboard,
    log_model_to_mlflow,
    log_params_to_mlflow,
    log_results_to_tensorboard,
    log_start_to_console,
    save_model,
)

LOG_MESSAGE_HANDLERS: messages.MessageHandlersConfig = {
    messages.LogStart: (log_start_to_console,),
    messages.LogMetrics: (
        log_metrics_to_console,
        log_metrics_to_tensorboard,
        log_metrics_to_mlflow,
    ),
    messages.LogEpoch: (log_epoch_to_console,),
    messages.LogBatch: (),
    messages.LogBatchStart: (log_batch_start_to_console,),
    messages.LogBatchEnd: (log_batch_end_to_console,),
    messages.LogResult: (log_results_to_tensorboard,),
    messages.LogParams: (log_params_to_mlflow,),
    messages.LogModel: (log_model_to_mlflow, save_model),
}

__all__ = ["LOG_MESSAGE_HANDLERS"]
