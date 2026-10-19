#!/usr/bin/env python3
"""
Обертка обратной совместимости для generate_features.py.
Перенаправляет выполнение в _tools/prepare_1s_dataset.py.
"""
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from _tools.prepare_1s_dataset import main

if __name__ == "__main__":
    main()
