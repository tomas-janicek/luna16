from .. import log_messages
from .ct_image_handlers import log_images_to_tensorboard
from .handlers import (
    log_batch_end_to_console,
    log_batch_start_to_console,
    log_epoch_to_console,
    log_input_to_mlflow,
    log_metrics_to_console,
    log_metrics_to_mlflow,
    log_metrics_to_tensorboard,
    log_model_to_mlflow,
    log_results_to_tensorboard,
    log_start_to_console,
)

LOG_MESSAGE_HANDLERS: log_messages.LogMessageHandlersConfig = {
    log_messages.LogStart: (log_start_to_console,),
    log_messages.LogMetrics: (
        log_metrics_to_console,
        log_metrics_to_tensorboard,
        log_metrics_to_mlflow,
    ),
    log_messages.LogEpoch: (log_epoch_to_console,),
    log_messages.LogBatch: (),
    log_messages.LogBatchStart: (log_batch_start_to_console,),
    log_messages.LogBatchEnd: (log_batch_end_to_console,),
    log_messages.LogResult: (log_results_to_tensorboard,),
    log_messages.LogImages: (log_images_to_tensorboard,),
    log_messages.LogModel: (log_model_to_mlflow,),
    log_messages.LogInput: (log_input_to_mlflow,),
}

__all__ = ["LOG_MESSAGE_HANDLERS"]
