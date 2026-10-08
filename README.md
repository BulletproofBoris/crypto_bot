# PRL/USDT Trading Bot

## 📌 Описание
Этот проект предназначен для предсказания цены и динамики её движения с целью автоматизированной торговли криптовалютной парой **PRL/USDT**.
Он использует **TensorFlow (GPU, CUDA)** для ансамбля моделей машинного обучения, технический анализ через **TA-Lib / pandas-ta**, а также стандартные DS-инструменты (**NumPy, Pandas, scikit-learn**).

## 📂 Структура проекта
trader_test/
│── _core/ # служебные модули проекта
│── notebooks/ # Jupyter/VS Code notebooks
│── data/ # данные (git-игнорятся)
│── requirements-tf.txt # основные зависимости
│── requirements-freeze.txt # полный слепок среды
│── env-export.txt # переменные окружения


## ⚙️ Среда разработки

### 1. Система
- **WSL2 + Ubuntu 24.04**
- **Python 3.12**
- **CUDA Toolkit 12.6**  
- NVIDIA драйвер (Windows): **555.xx+**

### 2. Виртуальное окружение
Создаётся так:
python3.12 -m venv .venv-tf
source .venv-tf/bin/activate

### 3. Установка зависимостей
Полный слепок:
pip install -r requirements-TF.txt

### 4. Переменные окружения
Для стабильной работы TensorFlow (без XLA, с контролем памяти):
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_XLA=0
export CUDA_CACHE_MAXSIZE=2147483648

### 5. Проверка установки
python - <<'PY'
import tensorflow as tf
print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
PY

#### Ожидаемый вывод:
TF: 2.21.0-dev20250909
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

### 6. 🔄 Воспроизведение на новой машине
Установить WSL2 + Ubuntu 24.04
Установить Python 3.12
Создать виртуальное окружение (.venv-tf)
Установить зависимости из requirements-freeze.txt
Подключить переменные окружения из env-export.txt
Проверить nvidia-smi и работу GPU в TensorFlow