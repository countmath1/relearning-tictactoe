import os
import sys

# Cluster env may set HF_HOME to an unwritable path; tests need a writable cache.
_hf_home = os.environ.get("HF_HOME", "")
_default_hf = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
if not _hf_home:
    os.environ["HF_HOME"] = _default_hf
else:
    try:
        if not os.path.isdir(_hf_home):
            os.makedirs(_hf_home, exist_ok=True)
        elif not os.access(_hf_home, os.W_OK):
            os.environ["HF_HOME"] = _default_hf
    except OSError:
        os.environ["HF_HOME"] = _default_hf
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Running as `python3 models/test_pool.py` puts `models/` on sys.path first; add project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.checkers import get_legal_moves, initial_state

from models.pool import ModelPool

if __name__ == "__main__":
    pool = ModelPool(dry_run=True)
    pool.load_all()
    s = initial_state()
    legal = get_legal_moves(s, 1)
    move, lp = pool.generate_move(0, s, 1, legal)
    assert move in legal, f"Got {move}, expected one of {legal}"
    print("Pool dry_run test passed. Move:", move)
