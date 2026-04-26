"""
Backward-compatible PRO training entrypoint.

`train_ultimate_150.py` is the canonical script for PRO training.
This file is kept only to avoid branch/merge breakage for existing commands.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_ultimate_150 import main


if __name__ == "__main__":
    main()
