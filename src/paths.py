import pathlib
import uuid

CODE_ROOT = pathlib.Path(__file__).parent
REPO_ROOT = CODE_ROOT.parent

DATA_ROOT = REPO_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

SAMPLES_DIR = REPO_ROOT / "samples"

def random_checkpoint_dir():
    rand_dir = REPO_ROOT / "checkpoints" / str(uuid.uuid4())
    assert not rand_dir.exists()
    return str(rand_dir)
