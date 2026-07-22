from verl.trainer.ray_trainer import is_training_complete


def test_training_stops_exactly_at_max_steps() -> None:
    assert not is_training_complete(global_step=0, training_steps=1)
    assert is_training_complete(global_step=1, training_steps=1)
    assert is_training_complete(global_step=2, training_steps=1)