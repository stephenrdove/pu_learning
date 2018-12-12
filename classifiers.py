import numpy as np
from sklearn.svm import LinearSVC
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
from sklearn.metrics import f1_score,precision_score,recall_score
import scipy as sp

from create_data import Dataset,load_dataset

def make_vectors(dataset):
    if hasattr(dataset, 'RN'): # We have RNs
        RN_labels = np.zeros(dataset.RN.shape[0])
        P_labels = np.ones(dataset.P.shape[0])
        labels = np.concatenate((P_labels,RN_labels))
        vectors = np.concatenate((dataset.P,dataset.RN))
    else: # Testing SVM, use U and U_labels
        print("No RN - using U and U_labels")
        labels = dataset.U_labels
        vectors = dataset.U
    
    return vectors,labels

def create_svm(dataset):
    vectors,labels = make_vectors(dataset)
    svm = LinearSVC()
    svm.fit(vectors,labels)
    print("\tSVM accuracy:")
    
    return svm

def create_label_prop(dataset):
    vectors,labels = make_vectors(dataset)
    
    Q_labels = -1 * np.ones(dataset.Q.shape[0] + dataset.test_X.shape[0])
    labels = np.concatenate((labels,Q_labels))
    vectors = np.concatenate((vectors,dataset.Q,dataset.test_X))
    
    label_prop = LabelSpreading()
    label_prop.fit(vectors,labels)
    print("\tLabel Propogation accuracy:")

    return label_prop

def test_clf(clf,dataset,test_name,y=None,on_train=False):
    if on_train:
        labels = np.concatenate((dataset.U_labels,np.ones(dataset.P.shape[0])))
        vectors = np.concatenate((dataset.U,dataset.P))
    else:
        vectors = dataset.test_X
        labels = dataset.test_labels

    if not y:
        y = clf.predict(vectors)

    #scores = [np.round(f1_score(labels,y),2), np.round(precision_score(labels,y),2),
    #             np.round(recall_score(labels,y),2)]
    #dataset.add_scores_dict(scores,test_name)
    dataset.add_scores_dict(f1_score(labels,y),test_name)

    num_vectors = len(labels)

    accuracy = 100 * sum(y == labels) / num_vectors
    print('\t\tAccuracy: ', accuracy, '%')
    print('\t\tF1 Score: ', f1_score(labels,y))
    print('\t\tPrecision Score: ', precision_score(labels,y))
    print('\t\tRecall Score: ', recall_score(labels,y))

    return y
    


if __name__ == "__main__":
    #data = load_dataset('testing.data')
    data = Dataset()
    svm = create_svm(data)
    test_clf(svm,data,'')





