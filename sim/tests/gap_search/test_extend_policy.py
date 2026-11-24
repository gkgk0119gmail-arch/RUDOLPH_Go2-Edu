import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from extend_policy import extend  # noqa: E402


def act(sd, x):
    for i in (0, 2, 4):
        x = torch.nn.functional.elu(x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    return x @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]


def test_extended_policy_acts_like_the_old_one():
    ck = torch.load(ROOT / "weights/go2_posture_walk_model_1999.pt", map_location="cpu", weights_only=True)
    new = extend(ck, 1, 0.6)
    x = torch.randn(64, 50)
    h = torch.rand(64, 1)
    for net in ("actor_state_dict", "critic_state_dict"):
        assert torch.allclose(act(ck[net], x), act(new[net], torch.cat([x, h], 1)), atol=1e-5)
    assert ck["actor_state_dict"]["mlp.0.weight"].shape[1] == 50  # source untouched
    assert (new["actor_state_dict"]["distribution.std_param"] == 0.6).all()
