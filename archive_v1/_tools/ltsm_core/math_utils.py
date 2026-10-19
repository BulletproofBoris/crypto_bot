import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import math
import tensorflow as tf
import gc
import time

class MathTrendAnalyzer:
    @staticmethod
    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    @staticmethod
    def calculate_macro_trend(losses_array):
        if len(losses_array) < 15: 
            return None, None, None, None
        
        df = pd.DataFrame({'loss': losses_array})
        q1, q3 = df['loss'].quantile(0.25), df['loss'].quantile(0.75)
        valid_max = min(q3 + 1.5 * (q3 - q1), df['loss'].quantile(0.85))
        
        y_data_raw = np.array(losses_array)
        x_data = np.arange(1, len(y_data_raw) + 1)
        
        valid_mask = (y_data_raw <= valid_max)
        x_fit = x_data[valid_mask]
        y_fit_raw = y_data_raw[valid_mask]
        
        if len(x_fit) < 10: 
            return None, None, None, None

        obs_amp = max(1e-4, np.max(y_fit_raw) - np.min(y_fit_raw))
        amplitude_guess = obs_amp
        min_c = max(0.0, np.min(y_fit_raw) - (obs_amp * 2.0))
        max_a = max(1e-3, obs_amp * 1.5)

        free_bounds = (
            [1e-5, 1e-5, min_c], 
            [max_a, 2.0, np.max(y_fit_raw)]
        )
        
        bootstrap_a, bootstrap_b, bootstrap_c = [], [], []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(50):
                sample_size = max(5, int(len(x_fit) * 0.8))
                sample_indices = np.sort(np.random.choice(len(x_fit), size=sample_size, replace=False))
                
                x_boot = x_fit[sample_indices]
                y_raw_boot = y_fit_raw[sample_indices]
                y_cummin_boot = np.minimum.accumulate(y_raw_boot)
                
                p0 = [amplitude_guess, 0.05, max(min_c, np.min(y_cummin_boot) - 0.01)]
                
                try:
                    popt, _ = curve_fit(
                        MathTrendAnalyzer.exp_func, x_boot, y_cummin_boot, 
                        p0=p0, bounds=free_bounds, maxfev=2000
                    )
                    bootstrap_a.append(popt[0])
                    bootstrap_b.append(popt[1])
                    bootstrap_c.append(popt[2])
                except:
                    pass
                
        if len(bootstrap_c) < 5:
            return None, None, None, None
            
        true_a = np.median(bootstrap_a)
        true_b = np.median(bootstrap_b)
        true_c = np.median(bootstrap_c)
        
        q5_c = np.percentile(bootstrap_c, 5)
        q95_c = np.percentile(bootstrap_c, 95)
        uncertainty = q95_c - q5_c
        
        margin = 0.001 
        runs_needed = 0
        if true_a > 0 and true_b > 0:
            base = margin / true_a
            if 0 < base < 1:
                runs_needed = int(np.ceil(-np.log(base) / true_b))
        
        return true_c, q5_c, uncertainty, runs_needed

def find_max_physical_batch(create_model_fn, seq_len, n_features, start_batch=1024):
    print(f"\n🔍 [Hardware Test] Ищем потолок VRAM (Старт с батча {start_batch})...")
    
    batch_size = 2 ** math.floor(math.log2(start_batch))
    
    while batch_size >= 16:
        try:
            tf.keras.backend.clear_session()
            gc.collect()
            
            model = create_model_fn(seq_len, n_features, 1e-5)
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            
            dummy_x = tf.random.normal((batch_size, seq_len, n_features))
            dummy_y = tf.random.normal((batch_size, 3))
            model.train_on_batch(dummy_x, dummy_y)
            
            print(f"✅ Физический предел VRAM найден: батч {batch_size}")
            
            del model, dummy_x, dummy_y
            tf.keras.backend.clear_session()
            gc.collect()
            
            return batch_size
            
        except (tf.errors.ResourceExhaustedError, Exception) as e:
            print(f"⚠️ Батч {batch_size} вызвал ошибку. Сброс контекста...")
            tf.keras.backend.clear_session()
            gc.collect()
            batch_size //= 2
            time.sleep(1) # Пауза для GPU
            
    return 16

def get_adaptive_batch_config(num_samples, max_phys_batch, target_steps=50):
    ideal_batch = num_samples / target_steps
    math_batch = int(2 ** np.ceil(np.log2(ideal_batch)))
    math_batch = max(math_batch, 16) 
    
    if math_batch <= max_phys_batch:
        return math_batch, math_batch, 1
    else:
        accum_steps = int(max(1, math_batch // max_phys_batch))
        return math_batch, max_phys_batch, accum_steps