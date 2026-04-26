"""
PRO training entrypoint.

This script is a thin alias to the Ultimate DreamerV3 training pipeline so that
the repo has a clear "PRO model" command:

    python train/train_pro.py --steps 1000000 --device mps --batch-size 64
"""

import os
import sys

# Ensure project root is on sys.path when launched as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_ultimate_150 import main


if __name__ == "__main__":
    main()
