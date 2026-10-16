import copy
import importlib
from types import SimpleNamespace

import pytest

from miles.utils.eval_config import build_eval_dataset_configs

_LAUNCH_SETTINGS = dict(rollout_stop=["global"], rollout_stop_token_ids=[11], eval_min_new_tokens=3)
_DEFAULTS = dict(stop=["default"], stop_token_ids=[22], min_new_tokens=5)
_DATASET_SETTINGS = dict(stop=["dataset"], stop_token_ids=[33], min_new_tokens=7)
_EMPTY_SETTINGS = dict(stop=[], stop_token_ids=[], min_new_tokens=0)
_NULL_SETTINGS = dict(stop=None, stop_token_ids=None, min_new_tokens=None)
_CASES = [
    pytest.param({}, {}, dict(stop=["global"], stop_token_ids=[11], min_new_tokens=3), id="launch"),
    pytest.param(_DEFAULTS, {}, _DEFAULTS, id="defaults"),
    pytest.param(_DEFAULTS, _DATASET_SETTINGS, _DATASET_SETTINGS, id="dataset"),
    pytest.param(_DEFAULTS, _EMPTY_SETTINGS, _EMPTY_SETTINGS, id="explicit-empty"),
    pytest.param(_DEFAULTS, _NULL_SETTINGS, _NULL_SETTINGS, id="explicit-null"),
    pytest.param(_EMPTY_SETTINGS, {}, _EMPTY_SETTINGS, id="empty-defaults"),
]


@pytest.mark.parametrize("defaults,overrides,expected", _CASES)
def test_config_resolves_stopping_settings(defaults, overrides, expected):
    args = SimpleNamespace(**copy.deepcopy(_LAUNCH_SETTINGS))
    raw = [dict(name="toy", path="cached.jsonl", **overrides)]
    before = copy.deepcopy((vars(args), raw, defaults))

    [config] = build_eval_dataset_configs(args, raw, defaults)

    assert {key: getattr(config, key) for key in expected} == expected
    assert (vars(args), raw, defaults) == before


def test_config_leaves_unset_stopping_settings_unset():
    [config] = build_eval_dataset_configs(SimpleNamespace(), [dict(name="toy", path="cached.jsonl")], {})

    assert {key: getattr(config, key) for key in _NULL_SETTINGS} == _NULL_SETTINGS


def test_training_sampling_preserves_the_existing_helper_call():
    from miles.rollout.inference_rollout.inference_rollout_common import compute_sampling_params

    args = SimpleNamespace(**_LAUNCH_SETTINGS, rollout_skip_special_tokens=True)
    params = compute_sampling_params(args, temperature=0.7, top_p=1.0, top_k=-1, max_new_tokens=32)

    assert params["stop"] == args.rollout_stop
    assert params["stop_token_ids"] == args.rollout_stop_token_ids
    assert params["min_new_tokens"] is None


@pytest.mark.parametrize("variant", ["default", "legacy"])
@pytest.mark.parametrize("defaults,overrides,expected", _CASES)
async def test_eval_forwards_resolved_stopping_settings(monkeypatch, variant, defaults, overrides, expected):
    from miles.utils.types import Sample

    module = importlib.import_module(
        "miles.rollout.sglang_rollout"
        if variant == "legacy"
        else "miles.rollout.inference_rollout.inference_rollout_eval"
    )
    args = SimpleNamespace(
        **copy.deepcopy(_LAUNCH_SETTINGS),
        group_rm=False,
        hf_checkpoint="fake-checkpoint",
        apply_chat_template=False,
        chat_template_path=None,
        rollout_skip_special_tokens=True,
        sglang_router_policy="round_robin",
        eval_reward_key=None,
        reward_key=None,
    )
    [config] = build_eval_dataset_configs(
        args, [dict(name="toy", path="cached.jsonl", n_samples_per_eval_prompt=1, **overrides)], defaults
    )
    cache_key = config.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    cache = {cache_key: SimpleNamespace(samples=[Sample(prompt="hello")])}
    captured = []

    async def generate(_state_or_args, sample, *, sampling_params, evaluation):
        assert evaluation
        captured.append(copy.deepcopy(sampling_params))
        sample.reward = 1.0
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(module, "generate_and_rm", generate)
    if variant == "legacy":
        monkeypatch.setattr(module, "EVAL_PROMPT_DATASET", cache)
        await module.eval_rollout_single_dataset(args, 0, config)
    else:
        await module.eval_rollout_single_dataset(SimpleNamespace(args=args), config, cache)

    [params] = captured
    assert {key: params[key] for key in expected} == expected
    assert {key: getattr(args, key) for key in _LAUNCH_SETTINGS} == _LAUNCH_SETTINGS
