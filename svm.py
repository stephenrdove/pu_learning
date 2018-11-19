import numpy as np
from sklearn.svm import LinearSVC
import scipy as sp

from create_data import Dataset,load_dataset


def create_svm(dataset):
    if hasattr(dataset, 'RN'): # We have RNs
        RN_labels = np.ones(dataset.RN.shape[0]) * -1
        P_labels = np.ones(dataset.P.shape[0])
        labels = np.concatenate((P_labels,RN_labels))
        vectors = np.concatenate((dataset.P,dataset.RN))
    else: # Testing SVM, use U and U_labels
        labels = dataset.U_labels
        vectors = dataset.U

    svm = LinearSVC()
    svm.fit(vectors,labels)

    return svm

def test_svm(svm,dataset):
    labels = dataset.U_labels #np.ones(dataset.P.shape[0])
    vectors = dataset.U

    num_vectors = len(labels)

    y = svm.predict(vectors)
    print(y)
    accuracy = 100 * sum(y == labels) / num_vectors
    print(accuracy, '%')



if __name__ == "__main__":
    #data = load_dataset('testing.data')
    data = Dataset()
    svm = create_svm(data)
    test_svm(svm,data)





