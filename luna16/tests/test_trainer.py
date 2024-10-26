from luna16 import datasets, message_handler, training

from . import fakes


def test_trainer(
    fake_model: fakes.FakeModel,
    fake_message_handler: fakes.FakeMessageHandler,
    fake_data_module: datasets.DataModule[fakes.SimpleCandidate],
) -> None:
    trainer = training.Trainer(
        name="Test Trainer", version="0.0.0-test", logger=fake_message_handler
    )
    trainer.fit(epochs=2, model=fake_model, data_module=fake_data_module)

    assert len(fake_message_handler.requested_messages) == 4
    assert isinstance(
        fake_message_handler.requested_messages[0], message_handler.LogStart
    )
    assert isinstance(
        fake_message_handler.requested_messages[1], message_handler.LogEpoch
    )
    assert isinstance(
        fake_message_handler.requested_messages[2], message_handler.LogEpoch
    )
    assert isinstance(
        fake_message_handler.requested_messages[3], message_handler.LogModel
    )

    assert len(fake_model.requested_training_params) == 2
