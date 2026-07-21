import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import visualizer

IMG_DIR = ROOT / "examples" / "demo_dataset" / "images"
LABEL_DIR = ROOT / "examples" / "demo_dataset" / "labels"
OUT_DIR = ROOT / "tests" / "output"

@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def test_visualizer_runs_without_error():
    img_path = IMG_DIR / "good.jpg"
    label_path = LABEL_DIR / "good.txt"
    out_path = OUT_DIR / "output_vis.jpg"
    
    assert img_path.exists()
    assert label_path.exists()
    
    visualizer.visualize(str(img_path), str(label_path), str(out_path))
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0