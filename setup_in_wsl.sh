#!/usr/bin/env bash
set -euo pipefail

# === Настройки ===
PROJECT_NAME="trader_test"
TARGET_DIR="$HOME/$PROJECT_NAME"

echo "[1/5] Копирую проект в WSL домашний каталог..."
rm -rf "$TARGET_DIR"
cp -r "$(pwd)" "$TARGET_DIR"

cd "$TARGET_DIR"

echo "[2/5] Обновляю пакеты Ubuntu..."
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip build-essential git

echo "[3/5] Создаю окружение RAPIDS..."
python3.12 -m venv .venv-rapids
source .venv-rapids/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-rapids.txt || true
deactivate

echo "[4/5] Создаю окружение TensorFlow..."
python3.12 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements-tf.txt || true
pip install "tensorflow[and-cuda]==2.17.0"
deactivate

echo "[5/5] Готово!"
echo "Проект перенесён в: $TARGET_DIR"
echo "Включение окружений:"
echo "  source $TARGET_DIR/.venv-rapids/bin/activate"
echo "  source $TARGET_DIR/.venv-tf/bin/activate"
