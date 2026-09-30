import json
import numpy as np
from supertonicsynth.frontend import SupertonicFrontend


def make_indexer(path):
    path.write_text(json.dumps(list(range(128))))


def test_encode_with_language(tmp_path):
    path = tmp_path / "unicode_indexer.json"
    make_indexer(path)
    frontend = SupertonicFrontend(path)
    batch = frontend.encode("hello", "en")
    assert batch.text.startswith("<en>")
    assert batch.text_ids.dtype == np.int64
    assert batch.text_mask.shape == (1, 1, len(batch.text_ids))
