from support import *

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

np.set_printoptions(precision=4, suppress=True, linewidth=150)

boston = load_boston()
data = boston.data
#data = normalize(boston.data)
data = StandardScaler().fit_transform(data)
X = pd.DataFrame(data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]


print(X.columns)
print()


spear_I = spearmans_importances(X, y)

print(spear_I)
plot_importances(spear_I.iloc[:8], imp_range=(0,1), width=3,
                 title="Boston Spearman's R importances")
plt.tight_layout()
plt.savefig("../images/boston-features-spearmans.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


'''
X_ = StandardScaler().fit_transform(X)
pca = PCA(svd_solver='full')
pca.fit(X_)

#print(list(pca.explained_variance_))
print(pca.explained_variance_ratio_)

#print( cross_val_score(pca, X) )

print("PC1")
print( pca.components_[0,:] )
pca_I = np.argsort(np.abs(pca.components_[0, :]))
print(pca_I)
print(X.columns[pca_I])

print()
print("PC2")
print( pca.components_[1,:] )
pca_I = np.argsort(np.abs(pca.components_[1, :]))
print(pca_I)
print(X.columns[pca_I])


# CCA Canonical Correlation Analysis?
'''
