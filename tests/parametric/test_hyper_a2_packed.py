from copy import deepcopy

import pytest
import torch

from nam.models.parametric import HyperA2


@pytest.mark.parametrize("mode", ["low_rank", "full"])
def test_joint_execution_matches_independent_outputs_gradients_and_state(mode):
    torch.set_num_threads(1)
    model = HyperA2.init_from_config(
        dict(
            params=[dict(name="Gain", type="continuous", min=0, max=10, default=5)],
            hypernet=dict(mode=mode, rank=2, hidden_sizes=[4], selector={}),
        )
    ).double()
    with torch.no_grad():
        for branch in model._branches.values():
            for parameter in branch._hypernet.parameters():
                parameter.add_(0.005 * torch.randn_like(parameter))
    model.set_uniform_batch_params(True)
    joint = deepcopy(model)
    joint.set_uniform_batch_params(True)
    x = 0.1 * torch.randn(2, model.receptive_field + 32, dtype=torch.float64)
    params = torch.tensor([[3.0], [3.0]], dtype=torch.float64)
    expected = torch.stack(
        [model.branch(size)(x, params, pad_start=False) for size in model.sizes], dim=1
    )
    state = set(joint.state_dict())
    rng = torch.get_rng_state()
    actual = joint.forward_sizes(x, params)
    assert torch.equal(rng, torch.get_rng_state())
    assert state == set(joint.state_dict())
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, original), (other_name, packed) in zip(
        model.named_parameters(), joint.named_parameters()
    ):
        assert name == other_name
        assert (original.grad is None) == (packed.grad is None)
        if original.grad is not None:
            torch.testing.assert_close(
                packed.grad, original.grad, rtol=1e-7, atol=1e-12
            )
