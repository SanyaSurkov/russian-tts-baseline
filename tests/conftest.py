import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def fake_preprocessed(tmp_path_factory):
    """Shared session-scoped fake preprocessed dir.

    Populated with the minimum JSON metadata consumed by Dataset,
    VarianceAdaptor, and FastSpeech2. Individual tests may extend this
    dir with per-test subdirs (mel/, pitch/, energy/, duration/) and
    metadata files (train.txt, etc.).
    """
    d = tmp_path_factory.mktemp("fake_preprocessed")

    (d / "speakers.json").write_text(
        json.dumps({"spk0": 0, "spk1": 1}), encoding="utf-8"
    )
    (d / "stats.json").write_text(
        json.dumps(
            {
                "pitch": [-2.0, 4.0, 0.0, 1.0],
                "energy": [-1.5, 3.5, 0.0, 1.0],
            }
        ),
        encoding="utf-8",
    )

    return d
