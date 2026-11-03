

import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# 1. Load the dataset into a Pandas DataFrame
section("1. Dataset")
data = {
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics":  [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms":  [13, 15, 9, 17, 10, 14],
}
df = pd.DataFrame(data).set_index("Student")
print(df)

# 2. Dimensions and descriptive statistics
section("2. Dimensions and descriptive statistics")
print("Shape (students, subjects):", df.shape)
print(df.describe())

# 3. Standardize: z = (x - mean) / std  (sample std, ddof=1)
section("3. Standardized data")
X = df.values.astype(float)
mean = X.mean(axis=0)
std = X.std(axis=0, ddof=1)
Z = (X - mean) / std
print(pd.DataFrame(Z, index=df.index, columns=df.columns))
print("\nMeans of Z (~0):", Z.mean(axis=0))
print("Std of Z   (=1):", Z.std(axis=0, ddof=1))

# 4. Correlation matrix of the standardized data

section("4. Correlation matrix of the standardized data")
n = Z.shape[0]
corr_matrix = (Z.T @ Z) / (n - 1)
print(pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns))
print("\nMatches np.corrcoef:", np.allclose(corr_matrix, np.corrcoef(Z, rowvar=False)))
print("Matches np.cov     :", np.allclose(corr_matrix, np.cov(Z, rowvar=False)))

# 5. Eigenvalues and eigenvectors (eigh: matrix is symmetric)
section("5. Eigenvalues and eigenvectors")
eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
print("Eigenvalues (unsorted):", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# 6. Sort eigenvalues in descending order and reorder eigenvectors
section("6. Sorted eigenvalues and eigenvectors")
idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[idx]
eigenvectors_sorted = eigenvectors[:, idx]
print("Sorted eigenvalues:", eigenvalues_sorted)
print("Sum of eigenvalues (= number of variables):", eigenvalues_sorted.sum())
print(pd.DataFrame(eigenvectors_sorted, index=df.columns,
                   columns=[f"PC{i+1}" for i in range(len(idx))]))

# 7. Select the two eigenvectors with the largest eigenvalues
section("7. Selected eigenvectors (matrix W)")
W = eigenvectors_sorted[:, :2]
print(pd.DataFrame(W, index=df.columns, columns=["PC1", "PC2"]))

# 8. Project the standardized data onto the two components: Y = Z W
section("8. Projection onto the two principal components")
Y = Z @ W
print(pd.DataFrame(Y, index=df.index, columns=["PC1", "PC2"]))

# 9. Explained variance ratio and cumulative explained variance
section("9. Explained variance")
evr = eigenvalues_sorted / eigenvalues_sorted.sum()
cum_evr = np.cumsum(evr)
print(pd.DataFrame({
    "Eigenvalue": eigenvalues_sorted,
    "Explained variance ratio": evr,
    "Cumulative": cum_evr,
}, index=[f"PC{i+1}" for i in range(len(evr))]))





"""
Lab Session 2 - Part B: Standardized PCA with Scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Dataset
data = {
    "Student": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "Mathematics": [12, 15, 8, 17, 10, 14],
    "Statistics":  [11, 14, 9, 16, 12, 15],
    "Programming": [14, 16, 10, 18, 11, 13],
    "Algorithms":  [13, 15, 9, 17, 10, 14],
}
df = pd.DataFrame(data).set_index("Student")
X = df.values.astype(float)
n = X.shape[0]

# 10. Standardized PCA with sklearn: StandardScaler + PCA
#     (PCA only centers the data, so scaling must be done explicitly.)
section("10. Standardized PCA (StandardScaler + PCA)")
scaler = StandardScaler()
Z_sk = scaler.fit_transform(X)
print("Standardized data (StandardScaler):")
print(pd.DataFrame(Z_sk, index=df.index, columns=df.columns))

# 11. Reduce the dataset to two principal components
section("11. Reduction to two principal components")
pca = PCA(n_components=2)
Y_sk = pca.fit_transform(Z_sk)
print("Shape of the projected data:", Y_sk.shape)
print("\nPrincipal axes (components_.T, one column per PC):")
print(pd.DataFrame(pca.components_.T, index=df.columns, columns=["PC1", "PC2"]))

# 12. Transformed coordinates of the students
section("12. Transformed coordinates of the students")
Y_sk_df = pd.DataFrame(Y_sk, index=df.index, columns=["PC1", "PC2"])
print(Y_sk_df)

# 13. Explained variance ratio and cumulative explained variance
section("13. Explained variance")
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
print(pd.DataFrame({
    "Eigenvalue (explained_variance_)": pca.explained_variance_,
    "Explained variance ratio": evr,
    "Cumulative": cum_evr,
}, index=["PC1", "PC2"]))

# Full PCA (all 4 components) to show the complete variance table
pca_full = PCA().fit(Z_sk)
print("\nAll components:")
print(pd.DataFrame({
    "Eigenvalue": pca_full.explained_variance_,
    "Explained variance ratio": pca_full.explained_variance_ratio_,
    "Cumulative": np.cumsum(pca_full.explained_variance_ratio_),
}, index=[f"PC{i+1}" for i in range(4)]))

# Important note for Part C (comparison with the from-scratch version)

section("Note: scaling convention (ddof=0 vs ddof=1)")
factor = np.sqrt(n / (n - 1))
print(f"sqrt(n/(n-1)) = {factor:.4f}   n/(n-1) = {factor**2:.4f}")

# Using the sample std (ddof=1) as in Part A gives directly comparable values
Z_ddof1 = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
pca_ddof1 = PCA(n_components=2)
Y_ddof1 = pca_ddof1.fit_transform(Z_ddof1)
print("\nCoordinates with ddof=1 standardization (comparable to Part A):")
print(pd.DataFrame(Y_ddof1, index=df.index, columns=["PC1", "PC2"]))
print("\nEigenvalues (ddof=1):", pca_ddof1.explained_variance_)
print("Explained variance ratio:", pca_ddof1.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca_ddof1.explained_variance_ratio_))
print("\nCheck: Y_sklearn(ddof=0) / Y(ddof=1) =", np.round(np.abs(Y_sk / Y_ddof1).mean(axis=0), 4))