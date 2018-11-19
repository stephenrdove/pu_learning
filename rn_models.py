import numpy as np
from sklearn.naive_bayes import MultinomialNB

from create_data import Dataset

def nb_clf(dataset):
    vectors = np.concatenate((dataset.P,dataset.U))
    labels = np.ones(vectors.shape[0])
    labels[-dataset.U.shape[0]:] = -1 * labels[-dataset.U.shape[0]:]
    
    clf = MultinomialNB(alpha=.01)
    clf.fit(vectors, labels)
    pred = clf.predict(dataset.U)
    
    rn_indices = np.where(pred == -1)
    dataset.make_rn(dataset.U[rn_indices])
    print('NB: Reliable Negatives saved to dataset.RN')

def rns_clf(dataset, num_rns):
    indices = np.arange(dataset.U.shape[0])
    np.random.shuffle(indices)
    rn_indices = indices[:num_rns]

    dataset.make_rn(dataset.U[rn_indices])
    print('RNS (%d samples): Reliable Negatives saved to dataset.RN' % num_rns)




