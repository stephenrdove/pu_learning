import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import scipy as sp


categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)

#for label in newsgroups_train["target_names"]:
#    print(label)

#for datum in newsgroups_train.data[:10]:
#    print(datum)
#print(newsgroups_train.target[2])



vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)
X = vectorizer.fit_transform(newsgroups_train.data)
#print(X)

cx = sp.sparse.coo_matrix(X)

for i,j,v in zip(cx.row, cx.col, cx.data):
    #print("(%d, %d), %s" % (i,j,v))
    pass

#for doc in cx.row:
#    print(cx.col)


x_ar = X.toarray()
cx = sp.sparse.coo_matrix(x_ar)

for i,j,v in zip(cx.row, cx.col, cx.data):
    #print("(%d, %d), %s" % (i,j,v))
    pass

print(x_ar[2033][9953])

#print(X.toarray())
#print(x_ar[2033][1549])
#x_again = sp.sparse.csr_matrix(x_ar)
#print(x_again)
#for (i,vector) in enumerate(X):
#    print()
#    print(vector[0])

print()
#print(X[(0, 29419)])

#print(newsgroups_train.target_names[newsgroups_train.target[0]])
#print(newsgroups_train.filenames[0])

#print(X.shape)
#print(X[0].shape)
#print(X.nnz / float(X.shape[0]))
'''
newsgroups_test = fetch_20newsgroups(subset='test',
                        remove=('headers', 'footers', 'quotes'),
                        categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(X, newsgroups_train.target)
pred = clf.predict(vectors_test)
print(metrics.f1_score(pred, newsgroups_test.target, average='macro'))
'''