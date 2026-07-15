from _core.tf_libs import tf, Model, Input, regularizers, Dense, Dropout, LSTM, Attention, LayerNormalization, EarlyStopping, ModelCheckpoint
# _core/project_setup.py

# --- 1. Настройка окружения и путей ---
import os
import sys
import warnings
import tensorflow as tf

#os.chdir('/content/drive/MyDrive/Colab Notebooks/trader_test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

try:
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, '_core')

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if CORE_DIR not in sys.path:
    sys.path.append(CORE_DIR)

# --- 2. Централизованные импорты сторонних библиотек ---
import argparse
import json
import math
import multiprocessing
import shutil
import time
import traceback
from datetime import datetime, date, timedelta
from functools import partial
from dateutil.relativedelta import relativedelta
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from scipy.signal import find_peaks
from tqdm import tqdm
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as mticker

print("✅ Окружение и сторонние библиотеки инициализированы.")