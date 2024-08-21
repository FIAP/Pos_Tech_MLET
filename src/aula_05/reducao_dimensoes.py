from sklearn.decomposition import TruncatedSVD, PCA, NMF
from scipy.sparse import csr_matrix


# Crie uma matriz esparsa com as dimensões desejadas
id_usuario = [1, 2, 3, 4]
id_item = [101, 102, 103, 104]
score_cliques = [0.5, 0.8, 0.2, 0.9]
score_similaridade = [0.3, 0.6, 0.1, 0.7]

# Converta as listas em uma matriz esparsa
matriz_esparsa = csr_matrix((score_cliques, (id_usuario, id_item)))

# Redução de dimensionalidade usando TruncatedSVD
svd = TruncatedSVD(n_components=2)
matriz_reduzida = svd.fit_transform(matriz_esparsa)
# Redução de dimensionalidade usando PCA

pca = PCA(n_components=2)
matriz_reduzida_pca = pca.fit_transform(matriz_esparsa)

# Redução de dimensionalidade usando Matrix Factorization (NMF)
nmf = NMF(n_components=2)
matriz_reduzida_nmf = nmf.fit_transform(matriz_esparsa)


# Comparativo de qualidade
print("TruncatedSVD:")
print(matriz_reduzida)
print("PCA:")
print(matriz_reduzida_pca)
print("NMF:")
print(matriz_reduzida_nmf)