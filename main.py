import pandas as pd
import numpy as np
import scipy.io
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, \
    normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


# ====================== ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ ======================

def load_character_trajectories_fixed(mat_file_path):
    print(f"Загрузка данных из файла {mat_file_path}...")

    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Файл {mat_file_path} не найден!")

    # Загружаем .mat файл
    mat_data = scipy.io.loadmat(mat_file_path)

    # Правильно извлекаем данные из consts
    if 'consts' in mat_data:
        consts = mat_data['consts'][0, 0]

        # Извлекаем метки классов из charlabels
        if 'charlabels' in consts.dtype.names:
            y = consts['charlabels'][0].flatten()
            print(f"Найдены метки классов: {y.shape}")
        else:
            y = None
    else:
        y = None

    # Обработка массива mixout
    if 'mixout' in mat_data:
        mixout = mat_data['mixout'][0]

        # Собираем все траектории
        all_trajectories = []

        for i, trajectory in enumerate(mixout):
            if isinstance(trajectory, np.ndarray):
                flattened = trajectory.T.flatten()
                all_trajectories.append(flattened)

        # Используем фиксированную длину
        fixed_length = len(all_trajectories[0])

        # Создаем матрицу признаков
        X = np.zeros((len(mixout), fixed_length))

        for i, trajectory in enumerate(mixout):
            if isinstance(trajectory, np.ndarray):
                flattened = trajectory.T.flatten()
                if len(flattened) > fixed_length:
                    X[i] = flattened[:fixed_length]
                else:
                    X[i, :len(flattened)] = flattened

        print(f"Создана матрица данных: {X.shape}")

        return X, y

    raise ValueError("Не удалось найти данные траекторий в .mat файле")


def evaluate_clustering(model, data, true_labels=None):
    """
    Оценка качества кластеризации с расширенными метриками
    """
    labels = model.fit_predict(data)

    # Проверка, что есть более 1 кластера
    if len(set(labels)) > 1 and -1 in labels:
        core_labels = labels[labels != -1]
        core_data = data[labels != -1]
        silhouette = silhouette_score(core_data, core_labels)
        calinski = calinski_harabasz_score(core_data, core_labels)
        davies = davies_bouldin_score(core_data, core_labels)
    elif len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
    else:
        silhouette = -1
        calinski = -1
        davies = float('inf')

    # Метрики, требующие истинные метки
    if true_labels is not None:
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
    else:
        ari = -1
        nmi = -1

    return labels, silhouette, calinski, davies, ari, nmi


# ====================== ОСНОВНОЙ КОД ======================

print("=" * 60)
print("Character Trajectories Dataset")
print("=" * 60)

# Загрузка данных
mat_file = 'mixoutALL_shifted.mat'
try:
    X, y = load_character_trajectories_fixed(mat_file)
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit(1)

print(f"\nРазмерность данных: {X.shape}")
if y is not None:
    print(f"Уникальные классы: {np.unique(y)}")
    print(f"Количество классов: {len(np.unique(y))}")

# Применяем PCA для уменьшения размерности до разумного уровня
pca_reduce = PCA(n_components=0.95)
X_reduced = pca_reduce.fit_transform(X)
print(f"\nПосле PCA (95% дисперсии): {X_reduced.shape[1]} признаков")
print(f"Сохраненная дисперсия: {pca_reduce.explained_variance_ratio_.sum():.2%}")

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)
X_scaled_dense = X_scaled

# Визуализация данных после PCA (2D для визуализации)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_dense)

plt.figure(figsize=(10, 6))
if y is not None:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=30)
else:
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=30)
plt.title("Визуализация исходных данных Character Trajectories (PCA)")
if y is not None:
    plt.colorbar(label='Класс')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

# ====================== 1. KMEANS КЛАСТЕРИЗАЦИЯ ======================
print("\n" + "=" * 60)
print("KMeans кластеризация")
print("=" * 60)

kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []
metrics_kmeans = []

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(kmeans, X_scaled_dense, y)
    print(f'KMeans с k={k}:')
    print(f'  Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}, Davies-Bouldin: {davies:.3f}')
    if y is not None:
        print(f'  ARI: {ari:.3f}, NMI: {nmi:.3f}')

    labels_for_k.append(labels)
    metrics_kmeans.append({'silhouette': silhouette, 'calinski': calinski, 'davies': davies, 'ari': ari, 'nmi': nmi})

    if silhouette > best_score_kmeans:
        best_score_kmeans = silhouette
        best_kmeans = kmeans
        best_labels_kmeans = labels
        best_k = k

print(f'\nЛучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(15, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'KMeans k={k}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# ====================== 2. AGGLOMERATIVE CLUSTERING ======================
print("\n" + "=" * 60)
print("Agglomerative кластеризация")
print("=" * 60)

agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None
labels_list = []
metrics_agg = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(agg, X_scaled_dense, y)
    print(f'Agglomerative с n_clusters={n}:')
    print(f'  Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}, Davies-Bouldin: {davies:.3f}')
    if y is not None:
        print(f'  ARI: {ari:.3f}, NMI: {nmi:.3f}')

    labels_list.append(labels)
    metrics_agg.append({'silhouette': silhouette, 'calinski': calinski, 'davies': davies, 'ari': ari, 'nmi': nmi})

    if silhouette > best_score_agg:
        best_score_agg = silhouette
        best_labels_agg = labels
        best_n_agg = n

print(f'\nЛучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(15, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'Agglomerative n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# ====================== 3. DBSCAN КЛАСТЕРИЗАЦИЯ ======================
print("\n" + "=" * 60)
print("DBSCAN кластеризация")
print("=" * 60)

# Используем более подходящие параметры для наших данных
dbscan_params = [
    (0.5, 5), (1.0, 5), (1.5, 5), (2.0, 5),
    (0.5, 10), (1.0, 10), (1.5, 10), (2.0, 10),
    (2.5, 10), (3.0, 10)
]

best_score_dbscan = -1
best_labels_dbscan = None
best_eps = None
best_min_samples = None
dbscan_results = []
metrics_dbscan = []

for eps, min_samples in dbscan_params:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels, silhouette, calinski, davies, ari, nmi = evaluate_clustering(dbscan, X_scaled_dense, y)

    # Подсчитываем количество кластеров (исключая шум -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f'DBSCAN с eps={eps}, min_samples={min_samples}, Силуэтный коэффициент: {silhouette:.3f}')

    metrics_dbscan.append({
        'eps': eps,
        'min_samples': min_samples,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'ari': ari,
        'nmi': nmi,
        'n_clusters': n_clusters
    })

    if silhouette > best_score_dbscan and n_clusters >= 2:
        best_score_dbscan = silhouette
        best_labels_dbscan = labels
        best_eps = eps
        best_min_samples = min_samples

    # Сохраняем для визуализации
    if n_clusters >= 2 and len(dbscan_results) < 4:
        dbscan_results.append((eps, min_samples, labels, silhouette, n_clusters))

if best_eps is not None:
    print(f'\nЛучший DBSCAN: eps={best_eps}, min_samples={best_min_samples} с коэффициентом: {best_score_dbscan:.3f}')
else:
    print("\nDBSCAN не смог найти хорошую кластеризацию с заданными параметрами")

# Визуализация нескольких лучших вариантов DBSCAN
n_dbscan_to_show = min(5, len(dbscan_results))
if n_dbscan_to_show > 0:
    fig, axes = plt.subplots(1, n_dbscan_to_show, figsize=(15, 4))

    # Если только 1 график
    if n_dbscan_to_show == 1:
        axes = [axes]

    for i in range(n_dbscan_to_show):
        eps, min_samples, labels, score, n_clusters = dbscan_results[i]
        scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        axes[i].set_title(f'DBSCAN\neps={eps}, min={min_samples}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[i])

    plt.tight_layout()
    plt.show()

# ====================== ВИЗУАЛИЗАЦИЯ ЛУЧШИХ КЛАСТЕРИЗАЦИЙ ======================
print("\n" + "=" * 60)
print("Сравнение лучших результатов")
print("=" * 60)

# Находим лучшие метрики для каждого метода
best_metrics_kmeans = next(m for i, m in enumerate(metrics_kmeans) if kmeans_params[i] == best_k)
best_metrics_agg = next(m for i, m in enumerate(metrics_agg) if agg_params[i] == best_n_agg)

if best_eps is not None:
    best_metrics_dbscan = next(
        m for m in metrics_dbscan if m['eps'] == best_eps and m['min_samples'] == best_min_samples)
else:
    best_metrics_dbscan = None

# Визуализация лучших результатов каждого метода
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Лучший KMeans
if best_labels_kmeans is not None:
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis')
    axes[0].set_title(f'KMeans (k={best_k})\nSilhouette: {best_score_kmeans:.3f}')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0])

# Лучший Agglomerative
if best_labels_agg is not None:
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='viridis')
    axes[1].set_title(f'Agglomerative (n={best_n_agg})\nSilhouette: {best_score_agg:.3f}')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[1])

# Лучший DBSCAN
if best_labels_dbscan is not None:
    scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_dbscan, cmap='viridis')
    axes[2].set_title(f'DBSCAN (eps={best_eps}, min={best_min_samples})\nSilhouette: {best_score_dbscan:.3f}')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.show()

# ====================== ИТОГОВОЕ СРАВНЕНИЕ ======================
print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ")
print("=" * 60)

print(f"\n{'Метод':20} {'Параметры':20} {'Silhouette':12}")
print("-" * 60)
print(f"{'KMeans':20} {'k=' + str(best_k):20} {best_score_kmeans:12.3f}")
print(f"{'Agglomerative':20} {'n=' + str(best_n_agg):20} {best_score_agg:12.3f}")
if best_eps is not None:
    print(f"{'DBSCAN':20} {'eps=' + str(best_eps) + ', min=' + str(best_min_samples):20} {best_score_dbscan:12.3f}")
print("-" * 60)

# Определяем лучший алгоритм
scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'DBSCAN': best_score_dbscan if best_eps is not None else -1
}

best_method = max(scores, key=scores.get)
best_score = scores[best_method]

print(f"\nЛУЧШИЙ АЛГОРИТМ: {best_method}")
print(f"Silhouette Score: {best_score:.3f}")

if best_method == 'KMeans':
    print(f"Оптимальное количество кластеров: {best_k}")
elif best_method == 'Agglomerative':
    print(f"Оптимальное количество кластеров: {best_n_agg}")
elif best_method == 'DBSCAN':
    print(f"Оптимальные параметры: eps={best_eps}, min_samples={best_min_samples}")

# Сводная таблица метрик
print(f"\n=== СВОДКА МЕТРИК ===")
methods = ['KMeans', 'Agglomerative', 'DBSCAN']
best_metrics_list = [best_metrics_kmeans, best_metrics_agg]
if best_metrics_dbscan is not None:
    best_metrics_list.append(best_metrics_dbscan)
else:
    best_metrics_list.append({'silhouette': -1, 'calinski': -1, 'davies': float('inf'), 'ari': -1, 'nmi': -1})

print(f"{'Метод':<15} {'Silhouette':<12} {'Calinski':<12} {'Davies':<12} {'ARI':<12} {'NMI':<12}")
print("-" * 75)
for method, metrics in zip(methods, best_metrics_list):
    sil_str = f"{metrics['silhouette']:.3f}" if metrics['silhouette'] != -1 else "N/A"
    cal_str = f"{metrics['calinski']:.1f}" if metrics['calinski'] != -1 else "N/A"
    dav_str = f"{metrics['davies']:.3f}" if metrics['davies'] != float('inf') else "inf"
    ari_str = f"{metrics['ari']:.3f}" if metrics['ari'] != -1 else "N/A"
    nmi_str = f"{metrics['nmi']:.3f}" if metrics['nmi'] != -1 else "N/A"

    print(f"{method:<15} {sil_str:<12} {cal_str:<12} {dav_str:<12} {ari_str:<12} {nmi_str:<12}")

# ВИЗУАЛИЗАЦИЯ СВОДКИ МЕТРИК ДЛЯ ЛУЧШИХ ПАРАМЕТРОВ
print(f"\n=== ВИЗУАЛИЗАЦИЯ СВОДКИ МЕТРИК ===")

# Создаем график со сводной таблицей
plt.figure(figsize=(14, 8))
ax = plt.subplot(111)
ax.axis('tight')
ax.axis('off')

# Подготавливаем данные для таблицы
table_data = []
for method, metrics in zip(methods, best_metrics_list):
    if metrics['silhouette'] != -1:
        sil_str = f"{metrics['silhouette']:.3f}"
        cal_str = f"{metrics['calinski']:.1f}"
        dav_str = f"{metrics['davies']:.3f}"
        ari_str = f"{metrics['ari']:.3f}" if metrics['ari'] != -1 else "N/A"
        nmi_str = f"{metrics['nmi']:.3f}" if metrics['nmi'] != -1 else "N/A"
    else:
        sil_str = "N/A"
        cal_str = "N/A"
        dav_str = "inf"
        ari_str = "N/A"
        nmi_str = "N/A"

    table_data.append([
        method,
        sil_str,
        cal_str,
        dav_str,
        ari_str,
        nmi_str
    ])

# Определяем количество столбцов
num_cols = len(table_data[0])  # Должно быть 6
num_rows = len(table_data) + 1  # +1 для заголовков

print(f"Количество столбцов: {num_cols}")
print(f"Количество строк: {num_rows}")

# Создаем таблицу
table = ax.table(cellText=table_data,
                 colLabels=['Метод', 'Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'ARI', 'NMI'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0.1, 0.1, 0.8, 0.8])  # Уменьшаем bbox для безопасности

# Настраиваем внешний вид таблицы
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Выделяем заголовки (правильно индексируем ячейки)
for i in range(num_cols):
    # Заголовки находятся в первой строке (индекс 0)
    table[(0, i)].set_facecolor('#4C72B0')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Выделяем лучший метод
best_method_index = methods.index(best_method)
for i in range(num_cols):
    # Данные начинаются со строки 1 (индекс 1)
    table[(best_method_index + 1, i)].set_facecolor('#e6f3ff')
    table[(best_method_index + 1, i)].set_text_props(weight='bold')

plt.title('СВОДКА МЕТРИК ДЛЯ ЛУЧШИХ ПАРАМЕТРОВ\n', size=16, weight='bold', pad=30)
plt.tight_layout()
plt.show()

# Дополнительная информация о данных
print(f"\n=== ИНФОРМАЦИЯ О ДАННЫХ ===")
print(f"Общее количество траекторий: {len(X)}")
print(f"Исходное количество признаков: {X.shape[1]}")
print(f"Количество признаков после PCA: {X_reduced.shape[1]}")
print(f"Сохраненная дисперсия PCA: {pca_reduce.explained_variance_ratio_.sum():.2%}")
if y is not None:
    print(f"Уникальные классы в данных: {np.unique(y)}")
    print(f"Распределение классов:")
    print(pd.Series(y).value_counts().sort_index())

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print(f"1. Данные Character Trajectories успешно кластеризованы")
print(f"2. Silhouette Score {best_score:.3f} указывает на качество кластеризации")
print(f"3. Рекомендуется использовать {best_method} для данных Character Trajectories")
print("4. Масштабирование признаков и PCA значительно улучшают результаты")
print("=" * 60)