import asyncio
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from miles.rollout import rm_hub
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.async_utils import run
from miles.utils.types import Sample


@pytest.fixture
def mock_args():
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = None
    args.rm_url = None
    return args


class TestAsyncRm:
    @pytest.mark.parametrize("fail", [False, True])
    def test_remote_rm_reuses_connections(self, mock_args, fail):
        async def check():
            transports = set()
            requests = []

            async def redirect(request):
                assert not request.cookies
                response = web.HTTPTemporaryRedirect("/reward")
                response.set_cookie("sample", "current")
                raise response

            async def reward(request):
                transports.add(request.transport)
                requests.append(await request.json())
                assert request.cookies["sample"] == "current"
                return web.json_response(1.0, status=503 if fail and len(requests) == 2 else 200)

            app = web.Application()
            app.router.add_post("/", redirect)
            app.router.add_post("/reward", reward)
            # IP hosts make aiohttp ignore cookies, masking cookie leakage between samples.
            async with TestServer(app, host="localhost") as server:
                mock_args.rm_type = "remote_rm"
                mock_args.rm_url = str(server.make_url("/"))
                sample = Sample(prompt="prompt", response="response", label="label")
                resume = asyncio.Event()

                async def after_pool_closes():
                    await resume.wait()
                    return await async_rm(mock_args, sample)

                async with rm_hub.remote_rm_connection_pool():
                    connector = rm_hub._remote_rm_pool.get()[1]
                    assert await async_rm(mock_args, sample) == 1.0
                    if fail:
                        with pytest.raises(aiohttp.ClientResponseError, match="503"):
                            await async_rm(mock_args, sample)
                    else:
                        assert await async_rm(mock_args, sample) == 1.0
                    assert await async_rm(mock_args, sample) == 1.0
                    assert len(transports) == 1
                    assert await asyncio.to_thread(lambda: asyncio.run(async_rm(mock_args, sample))) == 1.0
                    pending = asyncio.create_task(after_pool_closes())
                assert connector.closed
                assert rm_hub._remote_rm_pool.get() is None
                resume.set()
                assert await pending == 1.0
                assert len(transports) == 3
                assert requests == [{"prompt": "prompt", "response": "response", "label": "label"}] * 5

        run(check())

    def test_remote_rm_closes_pending_request_on_error(self, mock_args):
        async def check():
            started = asyncio.Event()
            disconnected = asyncio.Event()

            async def reward(request):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    disconnected.set()

            app = web.Application()
            app.router.add_post("/", reward)
            async with TestServer(app) as server:
                mock_args.rm_type = "remote_rm"
                mock_args.rm_url = str(server.make_url("/"))
                with pytest.raises(ValueError, match="rollout failed"):
                    async with rm_hub.remote_rm_connection_pool():
                        pending = asyncio.create_task(async_rm(mock_args, Sample(prompt="")))
                        await asyncio.wait_for(started.wait(), timeout=1)
                        raise ValueError("rollout failed")
                with pytest.raises(aiohttp.ClientError):
                    await asyncio.wait_for(pending, timeout=1)
                await asyncio.wait_for(disconnected.wait(), timeout=1)

        run(check())

    @pytest.mark.parametrize(
        "rm_type,response,label,expected",
        [
            ("math", r"\boxed{42}", "42", 1),
            (
                "math",
                r"<|open|>think<|sep|>work<|close|>think<|open|>response<|sep|>Answer: \boxed{42}<|close|>response",
                "42",
                1,
            ),
            ("math", r"\boxed{wrong}", "42", 0),
            ("f1", "hello world", "hello world", 1.0),
            ("dapo", "Answer: 42", "42", {"score": 1.0}),
            ("deepscaler", r"</think>\boxed{42}", "42", 1),
            ("gpqa", "Answer: A", "A", 1.0),
            ("boxed_f1", r"Final answer is \boxed{hello world}", "hello world", 1.0),
        ],
    )
    def test_rm_types(self, mock_args, rm_type, response, label, expected):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response=response, label=label)
        reward = run(async_rm(mock_args, sample))
        if isinstance(expected, dict):
            for k, v in expected.items():
                assert reward[k] == v
        else:
            assert reward == expected

    def test_f1_rm_partial(self, mock_args):
        mock_args.rm_type = "f1"
        sample = Sample(prompt="", response="hello", label="hello world")
        reward = run(async_rm(mock_args, sample))
        assert 0 < reward < 1

    def test_random_rm(self, mock_args):
        mock_args.rm_type = "random"
        sample = Sample(prompt="", response="anything", label="anything")
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_returns_binary(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello", label="", tokens=[1, 2, 3])
        reward = run(async_rm(mock_args, sample))
        assert reward in [0, 1]

    def test_deterministic_random_rm_is_deterministic(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        sample = Sample(prompt="", response="hello world", label="", tokens=[10, 20])
        rewards = [run(async_rm(mock_args, sample)) for _ in range(5)]
        assert len(set(rewards)) == 1

    def test_deterministic_random_rm_differs_by_response(self, mock_args):
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response=f"response_{i}", label="", tokens=[1, 2, 3]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_deterministic_random_rm_differs_by_tokens(self, mock_args):
        """Same response with different tokens yields both reward values across many samples."""
        mock_args.rm_type = "deterministic_random"
        samples = [Sample(prompt="", response="same", label="", tokens=[i, i + 1, i + 2]) for i in range(20)]
        rewards = [run(async_rm(mock_args, s)) for s in samples]
        assert 0 in rewards and 1 in rewards

    def test_rm_type_from_metadata(self, mock_args):
        mock_args.rm_type = None
        sample = Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"})
        reward = run(async_rm(mock_args, sample))
        assert reward == 1

    @pytest.mark.parametrize(
        "rm_type,match",
        [
            ("unknown_type", "not implemented"),
            ("", "not specified"),
        ],
    )
    def test_invalid_rm_type_raises(self, mock_args, rm_type, match):
        mock_args.rm_type = rm_type
        sample = Sample(prompt="", response="test", label="test")
        with pytest.raises(NotImplementedError, match=match):
            run(async_rm(mock_args, sample))


class TestBatchedAsyncRm:
    @pytest.mark.parametrize(
        "rm_type,samples_data,expected",
        [
            (
                "math",
                [(r"\boxed{42}", "42"), (r"\boxed{100}", "100"), (r"\boxed{wrong}", "42")],
                [1, 1, 0],
            ),
            (
                "f1",
                [("hello world", "hello world"), ("different", "something else")],
                [1.0, 0],
            ),
        ],
    )
    def test_batched_rm(self, mock_args, rm_type, samples_data, expected):
        mock_args.rm_type = rm_type
        samples = [Sample(prompt="", response=r, label=label) for r, label in samples_data]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards == expected

    def test_inplace_set_reward_field(self, mock_args):
        mock_args.rm_type = "math"
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42"),
            Sample(prompt="", response=r"\boxed{100}", label="100"),
        ]
        result = run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))
        assert result is None
        assert samples[0].reward == 1
        assert samples[1].reward == 1

    def test_inplace_raises_on_existing_reward(self, mock_args):
        mock_args.rm_type = "math"
        samples = [Sample(prompt="", response=r"\boxed{42}", label="42", reward=0.5)]
        with pytest.raises(AssertionError, match="Overriding"):
            run(batched_async_rm(mock_args, samples, inplace_set_reward_field=True))

    def test_empty_samples(self, mock_args):
        mock_args.rm_type = "math"
        rewards = run(batched_async_rm(mock_args, []))
        assert rewards == []

    def test_mixed_rm_types_via_metadata(self, mock_args):
        mock_args.rm_type = None
        samples = [
            Sample(prompt="", response=r"\boxed{42}", label="42", metadata={"rm_type": "math"}),
            Sample(prompt="", response="hello", label="hello", metadata={"rm_type": "f1"}),
        ]
        rewards = run(batched_async_rm(mock_args, samples))
        assert rewards[0] == 1
        assert rewards[1] == 1.0
