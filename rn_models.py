import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn import metrics
from scipy import sparse

from create_data import Dataset

def nb_clf(dataset):
    vectors = np.concatenate((dataset.P,dataset.U))
    P_size = dataset.P.shape[0]
    labels = np.ones(vectors.shape[0])
    labels[-dataset.U.shape[0]:] = -1 * labels[-dataset.U.shape[0]:]
    
    clf = MultinomialNB(alpha=.01)
    clf.fit(vectors, labels)
    pred = clf.predict(dataset.U)
    
    rn_indices = np.where(pred == -1)
    rn_indices = np.asarray([index for index in rn_indices[0] if index > P_size]) - P_size
    
    dataset.make_rn(rn_indices)
    print('\nNB: Reliable Negatives saved to dataset.RN')

def rns_clf(dataset, num_rns):
    indices = np.arange(dataset.U.shape[0])
    np.random.shuffle(indices)
    rn_indices = indices[:num_rns]

    dataset.make_rn(rn_indices)
    print('\nRNS (%d samples): Reliable Negatives saved to dataset.RN' % num_rns)

def cluster_clf(dataset,n_clusters,threshold=0.95):
    vectors = np.concatenate((dataset.P,dataset.U))
    U_size, P_size = dataset.U.shape[0], dataset.P.shape[0]

    vectors = sparse.csr_matrix(vectors)
    labels = np.ones(vectors.shape[0])
    labels[-U_size:] = -1 * labels[-U_size:]

    real_labels = dataset.true_labels

    clf = create_clusters(vectors,n_clusters,real_labels)
    cluster_labels = clf.labels_
    rn_labels = cluster_rn_labels(cluster_labels,labels,n_clusters,threshold)

    rn_indices = [i for i,label in enumerate(cluster_labels) if label in rn_labels and i > P_size]
    rn_indices = np.asarray(rn_indices) - P_size
    dataset.make_rn(rn_indices)

    print('\nCluster (%d clusters): Reliable Negatives saved to dataset.RN' % n_clusters)

def create_clusters(vectors,n_clusters,labels):
    
    done = False
    while not done:
        clf = MiniBatchKMeans(n_clusters=n_clusters,init='k-means++',n_init=1,
            init_size=1000, batch_size=1000,)
        clf.fit(vectors)

        #print("Homogeneity: %0.3f" % metrics.homogeneity_score(real_labels, clf.labels_))
        #print("Completeness: %0.3f" % metrics.completeness_score(real_labels, clf.labels_))
        #print("V-measure: %0.3f" % metrics.v_measure_score(labels, clf.labels_))
        
        done = (metrics.v_measure_score(labels, clf.labels_) > 0.3)

    return clf

def cluster_rn_labels(cluster_labels,labels,n_clusters,threshold):

    cluster_dict = {}
    for k in range(n_clusters):
        cluster_dict[k] = [0,0]

    for i,label in enumerate(cluster_labels):
        cluster_dict[label][labels[i] == -1] += 1
    
    rn_labels = [k for k in range(n_clusters) if cluster_dict[k][1]/sum(cluster_dict[k]) > threshold]
    if len(rn_labels) == 0:
        k_max = np.argmax([cluster_dict[k][1] for k in cluster_dict])
        rn_labels.append(k_max)
    
    return rn_labels
        




        




