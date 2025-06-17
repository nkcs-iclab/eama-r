from sklearn.cluster import AgglomerativeClustering
import numpy as np
import faiss


def kmeans(ncluster, embeddings, niter=10000):
    """
    kmeans聚类
    :param ncluster: 聚类数量
    :param embeddings: 输入embedding
    :param niter: 迭代niter次
    :return:
    """
    embeddings = np.asarray(embeddings)
    # niter = embeddings.shape[0] // ncluster + 20
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, ncluster, niter=niter, verbose=False)
    kmeans.train(embeddings)
    _, clusters = kmeans.index.search(embeddings, 1)
    centroids = kmeans.centroids
    return centroids, clusters, ncluster


def agglomerative(distance_threshold, embeddings):
    """
    层次聚类
    :param distance_threshold: 利用距离进行聚类
    :param embeddings: 输入embedding
    :return:
    """
    clustering_model = AgglomerativeClustering(n_clusters=None, compute_full_tree=True,
                                               linkage='average',
                                               distance_threshold=distance_threshold)
    embeddings = np.asarray(embeddings)
    clusters = clustering_model.fit_predict(embeddings)
    cluster_points = [embeddings[clusters == i] for i in range(clustering_model.n_clusters_)]

    # 计算每个聚类的中心
    centroids = [np.mean(points, axis=0) for points in cluster_points]
    ncluster = clustering_model.n_clusters_
    print(f"input {embeddings.shape[0]} embeddings, got {ncluster} clusters.")
    return centroids, clusters, ncluster
