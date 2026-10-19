# _core/tf_libs.py
import os, sys

# --- ИСПРАВЛЕНО: Добавлена переменная для отключения oneDNN ---
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' 
# -----------------------------------------------------------

# Тише до импорта
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")

# Заглушаем самый ранний stderr во время импорта TF
__stderr_backup = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    import tensorflow as tf
finally:
    try:
        sys.stderr.close()
    except Exception:
        pass
    sys.stderr = __stderr_backup

# Настраиваем GPU и логгеры после импорта
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.get_logger().setLevel("ERROR")
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception as e:
    print(f"Предупреждение при настройке TensorFlow/GPU: {e}")

# Keras 3 импортируется напрямую
import keras
from keras import Model, Input, regularizers
from keras.layers import Dense, Dropout, LSTM, Attention, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint