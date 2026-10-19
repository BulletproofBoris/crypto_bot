import sqlite3
import time
import json
import os
import numpy as np
from pathlib import Path
from .math_utils import MathTrendAnalyzer

class SmartOrchestrator:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._create_tables()

    def _execute(self, query, params=(), fetch=False, max_retries=5):
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(self.db_path, timeout=30.0) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;") 
                    cur = conn.cursor()
                    cur.execute(query, params)
                    if fetch:
                        return cur.fetchall()
                    conn.commit()
                    return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(np.random.uniform(0.1, 0.5))
                else:
                    raise

    def get_saving_threshold(self, fold_name, arch, keep=3):
        """
        Возвращает худший val_loss среди топ-N лучших моделей КОНКРЕТНОЙ АРХИТЕКТУРЫ.
        """
        # Ищем вхождение названия архитектуры в JSON-строке hyperparams
        arch_search = f'%"{arch}"%' 
        
        rows = self._execute(
            """SELECT val_loss FROM runs 
               WHERE fold=? AND status='COMPLETED' AND val_loss IS NOT NULL 
               AND hyperparams LIKE ?
               ORDER BY val_loss ASC LIMIT ?""",
            (fold_name, arch_search, keep), fetch=True
        )
        if len(rows) < keep:
            return float('inf')
        return rows[-1][0]

    def _create_tables(self):
        self._execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                config TEXT, fold TEXT, hyperparams TEXT,
                val_loss REAL, val_acc REAL, avg_epoch_time REAL,
                overhead_time REAL, total_ttc REAL, status TEXT,
                session_id TEXT DEFAULT 'legacy'
            )
        """)
        # Пытаемся добавить колонку "на лету", если БД была создана старой версией кода
        try:
            self._execute("ALTER TABLE runs ADD COLUMN session_id TEXT DEFAULT 'legacy'")
        except:
            pass # Если колонка уже есть, просто игнорируем ошибку

        self._execute("CREATE TABLE IF NOT EXISTS folds_meta (fold_name TEXT PRIMARY KEY, best_loss REAL)")
        self._execute("""
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY, fold TEXT,
                remaining_runs INTEGER, last_seen REAL
            )
        """)

    def update_worker_heartbeat(self, worker_id, fold_name, remaining_runs):
        current_time = time.time()
        self._execute(
            "INSERT OR REPLACE INTO workers (worker_id, fold, remaining_runs, last_seen) VALUES (?, ?, ?, ?)",
            (worker_id, fold_name, remaining_runs, current_time)
        )
        self._execute("DELETE FROM workers WHERE last_seen < ?", (current_time - 600,))

    def remove_worker(self, worker_id):
        self._execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))

    def sync_with_filesystem(self, models_dir, fold_name, dataset_name, current_arch=None):
        total_files, added_count = 0, 0
        best_synced_loss = float('inf')

        for meta_file in Path(models_dir).glob("*.json"):
            total_files += 1
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                run_id = data.get("run_id")
                loss = data.get("metrics", {}).get("val_loss")
                acc = data.get("metrics", {}).get("val_acc", 0) / 100.0 
                ttc = data.get("training_stats", {}).get("training_time_seconds", 0.0)

                # Достаем архитектуру из файла
                file_arch = data.get("arch", meta_file.name.split('_')[0])

                # 🛡️ ГЛАВНЫЙ ФИКС: Игнорируем модели чужих архитектур!
                if current_arch and file_arch != current_arch:
                    continue

                if not run_id or loss is None: continue

                if not self._execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,), fetch=True):
                    # Эмулируем правильные гиперпараметры, чтобы база их распознавала
                    hyperparams_str = json.dumps({"arch": file_arch})
                    
                    self._execute(
                        """INSERT INTO runs 
                           (run_id, config, fold, hyperparams, val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status, session_id) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'COMPLETED', 'legacy')""",
                        (run_id, dataset_name, fold_name, hyperparams_str, loss, acc, 0.0, 0.0, ttc)
                    )
                    added_count += 1
                    best_synced_loss = min(best_synced_loss, loss)
            except Exception as e:
                print(f"⚠️ Ошибка чтения {meta_file.name}: {e}")

        if total_files > 0 and added_count > 0:
            print(f"🔄 [Синхронизация] Восстановлено {added_count} записей архитектуры '{current_arch}'.")
            rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
            current_db_best = rows[0][0] if rows else float('inf')

            if best_synced_loss < current_db_best:
                self._execute("INSERT OR REPLACE INTO folds_meta (fold_name, best_loss) VALUES (?, ?)", (fold_name, best_synced_loss))
                print(f"🌍 [Оркестратор] Глобальный рекорд БД обновлен: {best_synced_loss:.4f}")

    def get_history(self, fold_name):
        rows = self._execute(
            "SELECT val_loss FROM runs WHERE fold=? AND status='COMPLETED' AND val_loss IS NOT NULL ORDER BY rowid ASC",
            (fold_name,), fetch=True
        )
        return [r[0] for r in rows]

    def evaluate_potential(self, fold_name, worker_id, remaining_runs, current_run_index):
        self.update_worker_heartbeat(worker_id, fold_name, remaining_runs)
        
        meta_rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
        global_best = meta_rows[0][0] if meta_rows else float('inf')

        res = self._execute("SELECT SUM(remaining_runs), COUNT(worker_id) FROM workers WHERE fold=?", (fold_name,), fetch=True)
        pool_budget = res[0][0] if res and res[0][0] else 0
        active_workers = res[0][1] if res else 0

        if current_run_index == 1:
            return True, f"Прогрев воркера. В пуле: {active_workers} процесс(ов).", global_best

        losses = self.get_history(fold_name)
        if len(losses) < 15:
            return True, f"Сбор статистики пулом ({len(losses)}/15)...", global_best

        true_c, q5_c, uncertainty, runs_needed = MathTrendAnalyzer.calculate_macro_trend(losses)
        
        if true_c is None:
            return True, "Недостаточно данных для оценки тренда.", global_best
            
        if uncertainty < 0.005:
            if global_best <= q5_c:
                return False, f"ЦЕЛЬ ДОСТИГНУТА! Рекорд ({global_best:.4f}) пробил доверительный порог.", global_best
            elif runs_needed > pool_budget:
                return False, f"БЮДЖЕТ: Цель на ~{runs_needed} ране, но в пуле осталось {pool_budget}.", global_best
            else:
                return True, f"Тренд ясен (Асимптота: {true_c:.4f}). Достигнем через {runs_needed} ранов.", global_best
        else:
            return True, f"Неопределенность высокая ({uncertainty:.5f}).", global_best

    def register_run_start(self, run_id, config, fold, hyperparams):
        # 🌟 ЧИТАЕМ ИДЕНТИФИКАТОР РОЯ ИЗ БАША (Либо ставим 'manual')
        session_id = os.environ.get("SWARM_ID", "manual_run")
        self._execute(
            "INSERT INTO runs (run_id, config, fold, hyperparams, status, session_id) VALUES (?, ?, ?, ?, 'TRAINING', ?)",
            (run_id, config, fold, json.dumps(hyperparams), session_id)
        )

    def register_run_end(self, run_id, fold_name, val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status='COMPLETED'):
        self._execute("""
            UPDATE runs SET val_loss=?, val_acc=?, avg_epoch_time=?, overhead_time=?, total_ttc=?, status=? 
            WHERE run_id=?
        """, (val_loss, val_acc, avg_epoch_time, overhead_time, total_ttc, status, run_id))
        
        if val_loss is not None:
            rows = self._execute("SELECT best_loss FROM folds_meta WHERE fold_name=?", (fold_name,), fetch=True)
            if not rows or val_loss < rows[0][0]:
                self._execute("INSERT OR REPLACE INTO folds_meta (fold_name, best_loss) VALUES (?, ?)", (fold_name, val_loss))
                print(f"🌍 [Оркестратор] Новый глобальный рекорд БД: {val_loss:.4f}")

    def should_prune_model(self, fold_name, current_loss, threshold=2.0):
        losses = self.get_history(fold_name)
        if len(losses) < 5: return False
        mu, sigma = np.mean(losses), np.std(losses)
        if sigma == 0: return False
        
        z_score = (current_loss - mu) / sigma
        if z_score > threshold:
            print(f"\n🔪 [Z-Score Pruning] Loss {current_loss:.4f} аномально высок (Z={z_score:.2f}).")
            return True
        return False