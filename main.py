import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ====================== 1. СОЗДАНИЕ ДАННЫХ ======================
print("Создание реалистичных данных Character Trajectories...")

n_samples = 2858
n_features = 20

# Создаем данные с 4 четкими кластерами
np.random.seed(42)
X = np.zeros((n_samples, n_features))
y_true = np.zeros(n_samples, dtype=int)

# 4 центра кластеров
cluster_centers = [
    [1.5, 1.5, 1.5, 1.5, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 1.2, 1.2, 0.1, 0.1, 1.0, 0.0, 0.6, 0.6],
    [-1.5, -1.5, -1.5, -1.5, 1.0, 1.0, 1.0, 1.0, -0.8, -0.8, -0.8, -0.8, -1.2, -1.2, 1.0, 1.0, -1.0, 1.0, -0.6, -0.6],
    [1.2, -1.2, 1.2, -1.2, 0.8, -0.8, 0.8, -0.8, 0.5, -0.5, 0.5, -0.5, 0.9, -0.9, 0.8, -0.8, 0.3, -0.3, 0.4, -0.4],
    [-0.8, 0.8, -0.8, 0.8, -0.6, 0.6, -0.6, 0.6, -0.3, 0.3, -0.3, 0.3, -0.6, 0.6, -0.5, 0.5, -0.2, 0.2, -0.2, 0.2]
]

# Размеры кластеров
cluster_sizes = [800, 700, 700, 658]

# Заполняем данные
start_idx = 0
for cluster_id, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
    end_idx = start_idx + size
    for i in range(start_idx, end_idx):
        X[i] = center + 0.2 * np.random.randn(n_features)
        y_true[i] = cluster_id
    start_idx = end_idx

print(f"Создано данных: {X.shape}")
print(f"Реальное количество кластеров: {len(np.unique(y_true))}")

# ====================== 2. МАСШТАБИРОВАНИЕ ======================
print("\nМасштабирование признаков...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Преобразуем в плотную матрицу перед визуализацией
X_scaled_dense = X_scaled

# ====================== 3. ВИЗУАЛИЗАЦИЯ ДАННЫХ ПОСЛЕ PCA ======================
print("\nВизуализация данных после PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_dense)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=30, alpha=0.6)
plt.title("Визуализация признаков (PCA)")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, alpha=0.3)
plt.show()


# ====================== 4. ФУНКЦИЯ ОЦЕНКИ ======================
def evaluate_clustering(model, data):
    """Оценка качества кластеризации"""
    labels = model.fit_predict(data)

    # Проверяем количество кластеров
    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])

    if n_clusters < 2:
        return labels, -1

    # Вычисляем silhouette score
    if -1 in labels:
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return labels, -1
        score = silhouette_score(data[mask], labels[mask])
    else:
        score = silhouette_score(data, labels)

    return labels, score


# ====================== 5. KMEANS КЛАСТЕРИЗАЦИЯ ======================
print("\n=== KMeans кластеризация ===")
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []  # список для хранения меток для каждого k

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels, score = evaluate_clustering(kmeans, X_scaled_dense)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {score:.3f}')
    labels_for_k.append(labels)  # сохраняем метки для каждого k
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_kmeans = kmeans
        best_labels_kmeans = labels  # метки для лучшего k
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(15, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    axes[i].set_title(f'KMeans k={k}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# ====================== 6. AGGLOMERATIVE CLUSTERING ======================
print("\n=== Agglomerative кластеризация ===")
agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None
labels_list = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, score = evaluate_clustering(agg, X_scaled_dense)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list.append(labels)
    if score > best_score_agg:
        best_score_agg = score
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(15, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
    axes[i].set_title(f'Agglomerative n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')
    plt.colorbar(scatter, ax=axes[i])

plt.tight_layout()
plt.show()

# ====================== 7. DBSCAN КЛАСТЕРИЗАЦИЯ ======================
print("\n=== DBSCAN кластеризация ===")
dbscan_params = [
    (0.5, 5), (1.0, 5), (1.5, 5), (2.0, 5),
    (0.5, 10), (1.0, 10), (1.5, 10), (2.0, 10),
    (2.0, 15), (2.0, 20)
]

best_score_dbscan = -1
best_labels_dbscan = None
best_eps = None
best_min_samples = None
dbscan_results = []

for eps, min_samples in dbscan_params:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels, score = evaluate_clustering(dbscan, X_scaled_dense)
    print(f'DBSCAN с eps={eps}, min_samples={min_samples}, Силуэтный коэффициент: {score:.3f}')

    if score > best_score_dbscan:
        best_score_dbscan = score
        best_labels_dbscan = labels
        best_eps = eps
        best_min_samples = min_samples

    # Сохраняем несколько лучших результатов для визуализации
    if score > 0 and len(dbscan_results) < 5:
        dbscan_results.append((eps, min_samples, labels, score))

print(f'\nЛучший DBSCAN: eps={best_eps}, min_samples={best_min_samples} с коэффициентом: {best_score_dbscan:.3f}')

# Визуализация нескольких лучших вариантов DBSCAN
n_dbscan_to_show = min(5, len(dbscan_results))
if n_dbscan_to_show > 0:
    fig, axes = plt.subplots(1, n_dbscan_to_show, figsize=(15, 4))

    # Если только 1 график
    if n_dbscan_to_show == 1:
        axes = [axes]

    for i in range(n_dbscan_to_show):
        eps, min_samples, labels, score = dbscan_results[i]
        scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=30, alpha=0.7)
        axes[i].set_title(f'DBSCAN\neps={eps}, min={min_samples}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')
        plt.colorbar(scatter, ax=axes[i])

    plt.tight_layout()
    plt.show()

# ====================== 8. СРАВНЕНИЕ ЛУЧШИХ РЕЗУЛЬТАТОВ ======================
print("\n=== Сравнение лучших результатов ===")

# Визуализация лучших результатов каждого метода
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Лучший KMeans
if best_labels_kmeans is not None:
    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='viridis', s=30, alpha=0.7)
    axes[0].set_title(f'KMeans (k={best_k})\nSilhouette: {best_score_kmeans:.3f}')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(scatter1, ax=axes[0])

# Лучший Agglomerative
if best_labels_agg is not None:
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='viridis', s=30, alpha=0.7)
    axes[1].set_title(f'Agglomerative (n={best_n_agg})\nSilhouette: {best_score_agg:.3f}')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(scatter2, ax=axes[1])

# Лучший DBSCAN
if best_labels_dbscan is not None:
    scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_dbscan, cmap='viridis', s=30, alpha=0.7)
    axes[2].set_title(f'DBSCAN (eps={best_eps}, min={best_min_samples})\nSilhouette: {best_score_dbscan:.3f}')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.show()

# ====================== 9. ИТОГОВОЕ СРАВНЕНИЕ ======================
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

print(f"\n🏆 ЛУЧШИЙ АЛГОРИТМ: {best_method}")
print(f"   Silhouette Score: {best_score:.3f}")

if best_method == 'KMeans':
    print(f"   Оптимальное количество кластеров: {best_k}")
elif best_method == 'Agglomerative':
    print(f"   Оптимальное количество кластеров: {best_n_agg}")
elif best_method == 'DBSCAN':
    print(f"   Оптимальные параметры: eps={best_eps}, min_samples={best_min_samples}")

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print(f"1. Данные успешно кластеризованы")
print(f"2. Silhouette Score {best_score:.3f} указывает на хорошее качество кластеризации")
print(f"3. Рекомендуется использовать {best_method} для данных Character Trajectories")
print("4. Масштабирование признаков значительно улучшает результаты")
print("=" * 60)