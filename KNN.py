# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
#
# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
# print(x, y)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2003)
#
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(x_train, y_train)
#
#
# correct = np.count_nonzero( (clf.predict(x_test) == y_test) == True )
# print("Accuary is : %.3f" % ( (correct) / len(x_test)))



#交叉验证
# import numpy as np
from sklearn import datasets
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import KFold
#
iris = datasets.load_iris()
X = iris.data
y = iris.target
#
# ks = [1, 3, 5, 7, 9, 11, 13, 15]
#
# kf = KFold(n_splits=5, random_state=2001, shuffle=True)
#
# best_k = ks[0]
# best_score = 0
#
# for k in ks:
#     curr_score = 0
#     for train_index, valid_index in kf.split(X):
#         clf = KNeighborsClassifier(n_neighbors=k)
#         clf.fit(X[train_index], y[train_index])
#         curr_score = curr_score + clf.score(X[valid_index], y[valid_index])
#     avg_score = curr_score / 5
#     if avg_score > best_score:
#         best_k = k
#         best_score = avg_score
# print(best_score)
# print(best_k)


from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

paramters = {'n_neighbors': [1, 3, 5, 7, 11, 13, 15]}
knn = KNeighborsClassifier()

clf = GridSearchCV(knn, paramters, cv=5)
clf.fit(X, y)

print(clf.best_score_)
print(clf.best_params_)

