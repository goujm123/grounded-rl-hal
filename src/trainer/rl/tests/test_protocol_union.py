import torch
from tensordict import TensorDict

from verl.protocol import union_tensor_dict


def test_union_unlocks_locked_tensordict_before_adding_keys() -> None:
    left = TensorDict({"responses": torch.tensor([[1, 2]])}, batch_size=[1]).lock_()
    right = TensorDict({"old_log_probs": torch.tensor([[0.1, 0.2]])}, batch_size=[1])

    merged = union_tensor_dict(left, right)

    assert not merged.is_locked
    assert merged["responses"].tolist() == [[1, 2]]
    torch.testing.assert_close(merged["old_log_probs"], torch.tensor([[0.1, 0.2]]))