import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from create_data import Dataset,load_dataset
from rn_models import nb_clf,rns_clf
from svm import create_svm,test_svm




if __name__ == '__main__':
    data = Dataset()
    nb_clf(data)
    svm = create_svm(data)

    test_svm(svm,data)

    for i in 1000*np.arange(1,10):
        rns_clf(data,i)
        svm = create_svm(data)
        test_svm(svm,data)

