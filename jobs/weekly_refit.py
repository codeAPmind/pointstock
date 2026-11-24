"""Sunday job: refit the direction models (see premarket/refit.py).

    cd ~/openclaw_workspace/PointStock && ~/miniconda3/envs/premarket/bin/python jobs/weekly_refit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from premarket.refit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
