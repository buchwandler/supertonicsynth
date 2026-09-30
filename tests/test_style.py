import json
import numpy as np
from supertonicsynth.style import load_voice_style


def test_load_style(tmp_path):
    path = tmp_path / "M1.json"
    path.write_text(
        json.dumps(
            {
                "style_ttl": {"dims": [1, 2], "data": [[1, 2]]},
                "style_dp": {"dims": [1, 2], "data": [[3, 4]]},
            }
        )
    )
    style = load_voice_style(path)
    assert style.name == "M1"
    assert style.ttl.dtype == np.float32
    assert style.dp.shape == (1, 2)
