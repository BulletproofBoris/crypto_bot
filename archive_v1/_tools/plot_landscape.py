import h5py
import numpy as np
import plotly.graph_objects as go
import argparse
import warnings
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Отключаем предупреждения для чистого вывода
warnings.filterwarnings("ignore")

def plot_landscape(fold_path):
    fold_path = Path(fold_path)
    trajectories_dir = fold_path / "models" / "trajectories"

    if not trajectories_dir.exists():
        print(f"❌ Папка траекторий не найдена: {trajectories_dir}")
        return

    print(f"🔍 Сканирование папки: {trajectories_dir}")
    
    all_weights = []
    all_losses = []
    
    # 1. Сбор данных
    for h5_file in trajectories_dir.glob("*.h5"):
        try:
            # Используем swmr=True для чтения файлов, которые пишутся воркерами
            with h5py.File(h5_file, 'r', swmr=True) as f:
                if 'trajectory' in f:
                    w = f['trajectory']['weights'][:]
                    l = f['trajectory']['val_loss'][:]
                    all_weights.append(w)
                    all_losses.append(l)
                    print(f"✅ Успешно прочитан: {h5_file.name} (эпох: {len(l)})")
        except Exception as e:
            print(f"⏭️ Файл занят или ошибка: {h5_file.name}")
            continue

    if not all_weights:
        print("❌ Данные не найдены.")
        return

    weights_matrix = np.concatenate(all_weights, axis=0)
    losses = np.concatenate(all_losses, axis=0)

    # 2. Фильтрация выбросов (оставляем лучшие 95% эпох)
    threshold = np.percentile(losses, 95)
    mask = losses <= threshold
    weights_matrix = weights_matrix[mask]
    losses = losses[mask]
    
    print(f"🧹 Фильтрация: отсечено {np.sum(~mask)} выбросов. Осталось точек: {len(losses)}")

    # 3. PCA: Снижение размерности
    print("📊 Выполнение PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights_matrix)

    # 4. Поверхность и сглаживание
    print("📈 Построение и сглаживание поверхности...")
    xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 100)
    yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((coords[:, 0], coords[:, 1]), losses, (X, Y), method='linear')
    
    # Сглаживание шума (Gaussian Filter)
    Z_smooth = gaussian_filter(Z, sigma=0.0) 

    # 5. Визуализация
    print("🎨 Генерация 3D-графика...")
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z_smooth, 
        colorscale='Viridis_r', 
        opacity=1.0, # Максимальная непрозрачность
        lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.8, specular=0.5)
    )])

    fig.update_layout(
        title=f"География гиперповерхности: {fold_path.name}",
        scene=dict(xaxis_title='PCA Component 1', yaxis_title='PCA Component 2', zaxis_title='Validation Loss'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 6. Сохранение
    output_path = fold_path / "landscape_surface.html"
    fig.write_html(str(output_path))
    print(f"✅ Готово! Файл сохранен: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Визуализация ландшафта гиперпространства")
    parser.add_argument("--fold", type=str, required=True, help="Путь к папке фолда")
    args = parser.parse_args()
    
    plot_landscape(args.fold)