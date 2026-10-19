import time
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import os

C_RED = '\033[91m'
C_RESET = '\033[0m'

class ElasticPatienceProfiler(Callback):
    def __init__(self, orchestrator, fold_name, max_epochs, bonus_ratio=0.1, min_delta=0.001,
                 hard_prune_epoch=25,     # Эпоха включения гильотины
                 hard_prune_z=4.0,        # Жесткий предел в сигмах
                 max_z_score=2.0):        # Элитный предел в сигмах
        super().__init__()
        self.orchestrator = orchestrator
        self.fold_name = fold_name
        self.max_epochs = max_epochs
        self.epoch_times = []
        self.pruned = False
        self.pending_msgs = [] # Хранилище отложенных сообщений
        
        self.micro_patience = max(1, int(0.12 * max_epochs))
        self.macro_patience = max(3.0, float(0.3 * max_epochs))
        self.macro_bonus = bonus_ratio * self.micro_patience 
        self.min_delta = min_delta 
        
        self.micro_wait = 0
        self.local_best_loss = np.inf 
        self.run_start_time = 0
        self.epoch_start_time = 0
        self.avg_epoch_time = 0.0
        self.overhead_time = 0.0
        self.total_ttc = 0.0
        
        self.hard_prune_epoch = hard_prune_epoch
        self.hard_prune_z = hard_prune_z
        self.max_z_score = max_z_score
        
        # ========================================================
        # 🧠 ЭЛИТНАЯ МАТЕМАТИКА ДЛЯ Z-SCORE
        # ========================================================
        valid_losses = []
        try:
            import sqlite3
            db_path = self.orchestrator.db_path 
            
            if os.path.exists(db_path):
                with sqlite3.connect(db_path, timeout=10.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT val_loss FROM runs WHERE fold = ? AND val_loss IS NOT NULL", (self.fold_name,))
                    valid_losses = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"⚠️ Не удалось прочитать базу для Elite Pruning: {e}")

        if len(valid_losses) > 5:
            k = max(3, int(len(valid_losses) * 0.15))
            elite_losses = np.sort(valid_losses)[:k]
            
            self.mu = np.mean(elite_losses) 
            self.sigma = np.std(elite_losses) + 1e-6 
        else:
            self.mu = 1.15 
            self.sigma = 0.1
            
        print(f"📐 [Elite Pruning] Элитная цель (Асимптота): {self.mu:.4f} | Допустимый разброс: ±{self.sigma:.4f}")
        
    def on_train_begin(self, logs=None):
        self.run_start_time = time.time()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        # Выводим накопившиеся с прошлой эпохи сообщения
        for msg in self.pending_msgs:
            print(msg)
        self.pending_msgs.clear()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)
        
        current_loss = logs.get('val_loss')
        if current_loss is None: return
        
        if current_loss < self.local_best_loss - self.min_delta:
            self.local_best_loss = current_loss
            self.micro_wait = 0
            self.macro_patience = min(float(self.max_epochs), self.macro_patience + self.macro_bonus)
        else:
            self.micro_wait += 1

        guillotine_limit = self.mu + (self.hard_prune_z * self.sigma)
        
        if epoch >= self.hard_prune_epoch and self.local_best_loss > guillotine_limit:
            self.pending_msgs.append(f"\n{C_RED}🔪 [Hard Pruning] Эпоха {epoch}: Лучший Loss {self.local_best_loss:.4f} > лимита {guillotine_limit:.4f} ({self.hard_prune_z} сигм от элиты). Итерация убита.{C_RESET}")
            self.model.stop_training = True
            self.pruned = True
            return

        if self.micro_wait >= self.micro_patience:
            z_score = (self.local_best_loss - self.mu) / self.sigma
            
            if z_score > self.max_z_score:
                self.pending_msgs.append(f"\n{C_RED}🔪 [Elite Z-Score] Нет улучшений {self.micro_patience} эпох. ЛУЧШИЙ Loss {self.local_best_loss:.4f} вылетел из элитного коридора (Z={z_score:.2f} > {self.max_z_score}). Итерация убита.{C_RESET}")
                self.model.stop_training = True
                self.pruned = True
                return
            else:
                self.micro_wait = 0 
                
        if (epoch + 1) >= int(self.macro_patience):
            self.pending_msgs.append(f"\n⏳ [Early Stopping] Обучение остановлено. Эластичный лимит: {int(self.macro_patience)} эпох.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.total_ttc = time.time() - self.run_start_time
        clean_epochs = self.epoch_times[1:] if len(self.epoch_times) > 1 else self.epoch_times
        self.avg_epoch_time = float(np.mean(clean_epochs)) if clean_epochs else 0.0
        pure_compute_time = sum(self.epoch_times)
        self.overhead_time = max(0.0, self.total_ttc - pure_compute_time)
        
        # Печатаем последние сообщения, если цикл был принудительно разорван
        for msg in self.pending_msgs:
            print(msg)
        self.pending_msgs.clear()


class SmartBacktrackCallback(Callback):
    def __init__(self, best_weights_path, monitor_loss='val_loss', factor=0.5, patience=3, min_lr=1e-6, max_rollbacks=3):
        super().__init__()
        # 🛡️ ПРИНУДИТЕЛЬНО ФИКСИРУЕМ ПАРАМЕТРЫ (Защита от сбоев оркестратора)
        self.monitor_loss = monitor_loss
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_rollbacks = max_rollbacks
        self.best_weights_path = str(best_weights_path)
        
        self.wait = 0
        self.rollback_count = 0
        self.best_loss = np.inf 

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor_loss)
        if current_loss is None: return

        if current_loss < self.best_loss - 1e-4:
            self.best_loss = current_loss
            self.wait = 0
            self.rollback_count = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.rollback_count += 1
                
                if self.rollback_count >= self.max_rollbacks:
                    print(f"\n🛑 Лимит откатов LR исчерпан.")
                    self.model.stop_training = True
                    return

                # 1. ЧИТАЕМ СТАРЫЙ LR *ДО* ЗАГРУЗКИ ВЕСОВ!
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = max(old_lr * self.factor, self.min_lr)

                # 2. ВОССТАНАВЛИВАЕМ ВЕСА
                if os.path.exists(self.best_weights_path):
                    self.model.load_weights(self.best_weights_path)

                # 3. ХИРУРГИЧЕСКИЙ СБРОС ИНЕРЦИИ ADAM
                if hasattr(self.model.optimizer, 'variables'):
                    opt_vars = self.model.optimizer.variables() if callable(self.model.optimizer.variables) else self.model.optimizer.variables
                    for var in opt_vars:
                        if 'iter' not in var.name.lower() and 'learning_rate' not in var.name.lower():
                            var.assign(tf.zeros_like(var))

                # 4. НАКЛАДЫВАЕМ НОВЫЙ LR ПОВЕРХ ТОГО, ЧТО ВОССТАНОВИЛ KERAS
                self.model.optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                
                print(f"\n📉 [Backtrack] Откат к ЛУЧШИМ весам + СБРОС ADAM. Новый LR: {new_lr:.7f}")

class FullTrajectoryTracker(Callback):
    def __init__(self, filepath, compress_level=9):
        super().__init__()
        self.filepath = filepath
        self.compress_level = compress_level
        self.file = None
        
        # Ссылки на HDF5 датасеты
        self.ds_weights = None
        self.ds_loss = None
        self.ds_val_loss = None
        self.ds_acc = None
        self.ds_val_acc = None
        self.ds_lr = None

    def on_train_begin(self, logs=None):
        # Открываем файл в режиме 'a' (append/read-write). 
        # Если файла нет - он создастся. Если есть - просто откроется.
        self.file = h5py.File(self.filepath, 'a')
        
        # Получаем количество параметров модели для "заголовка"
        dummy_weights = np.concatenate([w.flatten() for w in self.model.get_weights()])
        num_params = len(dummy_weights)

        # === 1. ПРОВЕРКА И ЗАПИСЬ "ЗАГОЛОВКА" ===
        if 'metadata' not in self.file:
            # Создаем пустую группу для заголовка, если это первый запуск
            meta_group = self.file.create_group('metadata')
            meta_group.attrs['num_parameters'] = num_params
            meta_group.attrs['optimizer'] = self.model.optimizer.__class__.__name__
        else:
            # Если файл уже существует, проверяем, та ли это модель
            saved_params = self.file['metadata'].attrs['num_parameters']
            if saved_params != num_params:
                print(f"⚠️ ВНИМАНИЕ: Размерность модели ({num_params}) не совпадает с кэшем ({saved_params})! Возможен конфликт.")

        # === 2. ИНИЦИАЛИЗАЦИЯ ИЛИ ЗАГРУЗКА РЕЗИНОВЫХ МАССИВОВ ===
        if 'trajectory' not in self.file:
            traj_group = self.file.create_group('trajectory')
            
            # maxshape=(None, ...) означает, что массив может расти бесконечно
            self.ds_weights = traj_group.create_dataset(
                'weights', shape=(0, num_params), maxshape=(None, num_params), 
                dtype='float16', chunks=True, compression='gzip', compression_opts=self.compress_level
            )
            self.ds_loss = traj_group.create_dataset('loss', shape=(0,), maxshape=(None,), dtype='float32')
            self.ds_val_loss = traj_group.create_dataset('val_loss', shape=(0,), maxshape=(None,), dtype='float32')
            self.ds_acc = traj_group.create_dataset('acc', shape=(0,), maxshape=(None,), dtype='float32')
            self.ds_val_acc = traj_group.create_dataset('val_acc', shape=(0,), maxshape=(None,), dtype='float32')
            self.ds_lr = traj_group.create_dataset('learning_rate', shape=(0,), maxshape=(None,), dtype='float32')
        else:
            # Подхватываем существующие датасеты, чтобы дописывать в них
            traj_group = self.file['trajectory']
            self.ds_weights = traj_group['weights']
            self.ds_loss = traj_group['loss']
            self.ds_val_loss = traj_group['val_loss']
            self.ds_acc = traj_group['acc']
            self.ds_val_acc = traj_group['val_acc']
            self.ds_lr = traj_group['learning_rate']

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # 1. Извлекаем и сплющиваем веса
        raw_weights = self.model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in raw_weights]).astype(np.float16)

        # 2. Достаем текущий Learning Rate (он может меняться твоим SmartBacktrackCallback)
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # 3. Увеличиваем размер массивов на 1 строку
        current_size = self.ds_weights.shape[0]
        new_size = current_size + 1
        
        self.ds_weights.resize(new_size, axis=0)
        self.ds_loss.resize(new_size, axis=0)
        self.ds_val_loss.resize(new_size, axis=0)
        self.ds_acc.resize(new_size, axis=0)
        self.ds_val_acc.resize(new_size, axis=0)
        self.ds_lr.resize(new_size, axis=0)

        # 4. Записываем данные текущей эпохи в самый конец массива
        self.ds_weights[current_size] = flat_weights
        self.ds_loss[current_size] = logs.get('loss', 0.0)
        self.ds_val_loss[current_size] = logs.get('val_loss', 0.0)
        self.ds_acc[current_size] = logs.get('accuracy', logs.get('acc', 0.0))
        self.ds_val_acc[current_size] = logs.get('val_accuracy', logs.get('val_acc', 0.0))
        self.ds_lr[current_size] = current_lr

        # Принудительно сбрасываем буфер на диск, чтобы при краше ничего не потерять
        self.file.flush()

    def on_train_end(self, logs=None):
        if self.file:
            self.file.close()