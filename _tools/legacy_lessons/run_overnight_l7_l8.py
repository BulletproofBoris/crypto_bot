#!/usr/bin/env python3
"""
_tools/run_overnight_l7_l8.py
=============================================================================
Автоматический 8-часовой конвейер обучения:
  1) Урок 7: Triad Mastery (1 Normal + 1 Dump + 1 Pump, 525 итераций, ~3.9ч)
  2) Урок 8: Continuous 1800s Marathon & Capital Velocity (525 итераций, ~3.9ч)
=============================================================================
"""

import gc
import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from _tools.lesson_7_triad_mastery_trainer import run_lesson_7
from _tools.lesson_8_continuous_marathon_trainer import run_lesson_8


def main():
    run_lesson_7(iterations=525, n_groups=1365)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_lesson_8(iterations=525, n_groups=1365)


if __name__ == "__main__":
    main()
