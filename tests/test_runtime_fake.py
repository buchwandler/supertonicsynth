from types import SimpleNamespace
import json
import numpy as np
from supertonicsynth._onnxvoice import ResolvedSupertonicBundle
from supertonicsynth.runtime import SupertonicRuntime


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, token_ids, **kwargs):
        self.calls.append((token_ids, kwargs))
        return SimpleNamespace(audio=np.ones(100, dtype=np.float32) * 0.1, sample_rate=1000)

    def close(self):
        self.closed = True


def test_runtime_fake(tmp_path):
    indexer = tmp_path / "unicode_indexer.json"
    indexer.write_text(json.dumps(list(range(128))))
    config = tmp_path / "tts.json"
    config.write_text("{}")
    style = tmp_path / "M1.json"
    style.write_text(
        json.dumps(
            {
                "style_ttl": {"dims": [1, 2], "data": [[1, 2]]},
                "style_dp": {"dims": [1, 2], "data": [[3, 4]]},
            }
        )
    )
    bundle = ResolvedSupertonicBundle(
        None, "test", config, indexer, {}, {"M1": style}, 1000, None, {}
    )
    fake = FakeRuntime()
    runtime = SupertonicRuntime(fake, bundle)
    result = runtime.synthesize_text("hello", voice="M1", language="en", seed=4)
    assert result.sample_rate == 1000
    assert result.audio.size == 100
    assert fake.calls[0][1]["seed"] == 4
    runtime.close()
    assert fake.closed
