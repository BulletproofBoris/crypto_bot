#!/usr/bin/env python3
"""Фабрика нейросетевых архитектур, GPU-генераторов и коллбэков для обучения роя (Colab / Local)."""
import os
import time
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, Conv1D, GaussianNoise,
    GRU, Add, Activation, Flatten, GlobalAveragePooling1D,
    MultiHeadAttention, SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def make_gpu_sequence_dataset(X_gpu, y_gpu, seq_len=75, batch_size=2048, shuffle=True):
    X_gpu = tf.constant(X_gpu, dtype=tf.float16)
    y_gpu = tf.constant(y_gpu, dtype=tf.float32)
    offsets_gpu = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
    n_win = int(X_gpu.shape[0]) - seq_len + 1
    ds = tf.data.Dataset.from_tensor_slices(tf.range(n_win, dtype=tf.int32))
    if shuffle:
        ds = ds.shuffle(buffer_size=n_win, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(
        lambda b_idx: (tf.gather(X_gpu, b_idx[:, tf.newaxis] + offsets_gpu), tf.gather(y_gpu, b_idx)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds.prefetch(tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)

# 4. Фабрика 4 нейросетевых архитектур (из _tools/ltsm_core/model_builder.py)
def build_conv1d_gru(seq_len, n_features, n_classes=7, l2_reg=1e-4):
    inp = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inp)
    x = LayerNormalization()(x)
    x = Conv1D(32, 1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Conv1D(64, 3, padding='causal', activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)
    x = GRU(64, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.15)(x)
    out = Dense(n_classes, activation='softmax', dtype='float32')(x)
    return Model(inp, out, name="conv1d_gru")

def _res_block(x, filters, kernel_size, dilation, l2_reg):
    sc = x
    r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation, kernel_regularizer=regularizers.l2(l2_reg))(x)
    r = LayerNormalization()(r)
    r = Activation('gelu')(r)
    r = Dropout(0.1)(r)
    r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation, kernel_regularizer=regularizers.l2(l2_reg))(r)
    r = LayerNormalization()(r)
    if int(sc.shape[-1]) != filters:
        sc = Conv1D(filters, 1, padding='valid', kernel_regularizer=regularizers.l2(l2_reg))(sc)
    return Activation('gelu')(Add()([sc, r]))

def build_causal_cnn(seq_len, n_features, n_classes=7, l2_reg=1e-4):
    inp = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inp)
    x = LayerNormalization()(x)
    x = Conv1D(32, 1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = _res_block(x, 32, 3, 1, l2_reg)
    x = _res_block(x, 64, 3, 2, l2_reg)
    x = _res_block(x, 64, 3, 4, l2_reg)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(0.15)(x)
    out = Dense(n_classes, activation='softmax', dtype='float32')(x)
    return Model(inp, out, name="causal_cnn")

def build_attention(seq_len, n_features, n_classes=7, l2_reg=1e-4):
    inp = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inp)
    x = LayerNormalization()(x)
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.2)(x, x)
    x = LayerNormalization()(Add()([x, attn]))
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.15)(x)
    out = Dense(n_classes, activation='softmax', dtype='float32')(x)
    return Model(inp, out, name="attention")

def build_mlp_bottleneck(seq_len, n_features, n_classes=7, l2_reg=1e-4):
    inp = Input(shape=(seq_len, n_features), name="input_layer")
    x = GaussianNoise(0.01)(inp)
    x = LayerNormalization()(x)
    x = SpatialDropout1D(0.20)(x)
    x = Conv1D(32, 1, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(l2_reg * 3))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.20)(x)
    sc = x
    x = Dense(128, kernel_regularizer=regularizers.l2(l2_reg * 3))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.20)(x)
    x = Add()([sc, x])
    x = Dense(64, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.15)(x)
    out = Dense(n_classes, activation='softmax', dtype='float32')(x)
    return Model(inp, out, name="mlp_bottleneck")

MODEL_FACTORY = {
    "conv1d+gru": build_conv1d_gru,
    "causal_cnn": build_causal_cnn,
    "attention": build_attention,
    "mlp_bottleneck": build_mlp_bottleneck
}


class SmartBacktrackCallback(tf.keras.callbacks.Callback):
    """Откат к лучшим весам + хирургический сброс моментов AdamW при плато val_loss (из ltsm_core/callbacks.py)."""
    def __init__(self, best_weights_path, monitor_loss="val_loss", factor=0.5, patience=2, min_lr=1e-5, max_rollbacks=3):
        super().__init__()
        self.best_weights_path = str(best_weights_path)
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss) if logs else None
        if current_loss is None:
            return
        if current_loss < self.best_loss - 1e-4:
            self.best_loss = current_loss
            self.wait = 0
            self.rollback_count = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.rollback_count += 1
                if self.rollback_count >= self.max_rollbacks:
                    print(f"   🛑 [Backtrack] Лимит откатов LR ({self.max_rollbacks}) исчерпан.")
                    self.model.stop_training = True
                    return
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(old_lr * self.factor, self.min_lr)
                if os.path.exists(self.best_weights_path):
                    self.model.load_weights(self.best_weights_path)
                if hasattr(self.model.optimizer, "variables"):
                    opt_vars = self.model.optimizer.variables() if callable(self.model.optimizer.variables) else self.model.optimizer.variables
                    for var in opt_vars:
                        vname = var.name.lower()
                        if len(var.shape) >= 1 and any(k in vname for k in ("momentum", "velocity", "exponential_moving")):
                            var.assign(tf.zeros_like(var))
                self.model.optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                print(f"   📉 [Backtrack] Откат к лучшим весам + сброс инерции AdamW -> новый LR: {new_lr:.6f}")


class ElasticPatienceProfiler(tf.keras.callbacks.Callback):
    """Эластичный бюджет эпох + Z-Score отбраковка слабых инициализаций (масштабировано в 5 раз)."""
    def __init__(self, history_losses, max_epochs=10, bonus_ratio=0.25, min_delta=5e-4,
                 hard_prune_epoch=5, hard_prune_z=4.0, max_z_score=2.0):
        super().__init__()
        self.max_epochs = max_epochs
        self.pruned = False
        self.micro_patience = max(1, int(round(0.20 * max_epochs)))   # 2 эпохи
        self.macro_patience = max(4.0, float(0.45 * max_epochs))     # старт с 4.5 эпох, растет при успехах до 10
        self.macro_bonus = max(0.75, bonus_ratio * self.micro_patience)
        self.min_delta = min_delta
        self.micro_wait = 0
        self.local_best_loss = np.inf
        self.hard_prune_epoch = hard_prune_epoch
        self.hard_prune_z = hard_prune_z
        self.max_z_score = max_z_score

        if len(history_losses) >= 3:
            k = max(2, int(np.ceil(len(history_losses) * 0.35)))
            elite = np.sort(history_losses)[:k]
            self.mu = float(np.mean(elite))
            self.sigma = float(max(np.std(elite), 0.015))
        else:
            self.mu = 1.65
            self.sigma = 0.15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss") if logs else None
        if current_loss is None:
            return
        if current_loss < self.local_best_loss - self.min_delta:
            self.local_best_loss = current_loss
            self.micro_wait = 0
            self.macro_patience = min(float(self.max_epochs), self.macro_patience + self.macro_bonus)
        else:
            self.micro_wait += 1

        guillotine_limit = self.mu + self.hard_prune_z * self.sigma
        if (epoch + 1) >= self.hard_prune_epoch and self.local_best_loss > guillotine_limit:
            print(f"   🔪 [Hard Pruning] Эпоха {epoch+1}: лучший Loss {self.local_best_loss:.4f} > гильотины {guillotine_limit:.4f}. Итерация сброшена.")
            self.model.stop_training = True
            self.pruned = True
            return

        if self.micro_wait >= self.micro_patience and len( 
            [1] if self.mu < 1.60 else []
        ) > 0:
            z_score = (self.local_best_loss - self.mu) / self.sigma
            if z_score > self.max_z_score:
                print(f"   🔪 [Elite Z-Score] Нет прогресса {self.micro_patience} эп. Loss {self.local_best_loss:.4f} вне элиты (Z={z_score:.2f}). Итерация сброшена.")
                self.model.stop_training = True
                self.pruned = True
                return
            self.micro_wait = 0

        if (epoch + 1) >= int(np.ceil(self.macro_patience)):
            print(f"   ⏳ [Elastic EarlyStop] Достигнут эластичный потолок ({int(np.ceil(self.macro_patience))}/{self.max_epochs} эп.).")
            self.model.stop_training = True

# Инициализируем локальную SQLite БД оркестратора (как в SmartOrchestrator)


class GPUMonitorCallback(tf.keras.callbacks.Callback):
    """Мониторинг загрузки GPU и видеопамяти в конце эпохи."""
    def on_epoch_end(self, epoch, logs=None):
        if epoch in (0, 4, 9):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    text=True
                ).strip()
                print(f"    [GPU T4 Эпоха {epoch+1}] Загрузка/VRAM: {out}")
            except Exception:
                pass
