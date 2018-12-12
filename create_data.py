import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pickle


class Dataset():

    def __init__(self,category=0,gamma=0.5,cats=None):
        assert isinstance(category,int),"Category must be an int."
        assert gamma > 0 and gamma < 1,"gamma must be between 0 and 1"
        self.category = category
        self.cats = cats
        self.scores_dict = {}
        self.vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)
        self.train_data()
        self.make_pu(gamma)
        self.test_data()


    def train_data(self):
        self.full_train = fetch_20newsgroups(subset='train',
                        remove=('headers', 'footers', 'quotes'),
                        categories=self.cats)#,shuffle=True)
        self.full_X = self.vectorizer.fit_transform(self.full_train.data)
        self.train_vector_length = self.full_X.shape[1]
        self.full_labels = self.full_train.target
        X_ary = self.full_X.toarray()
        positive = []
        negative = []
        negative_true = []
        positive_true = []
        for i,x in enumerate(X_ary):
            if self.full_train.target[i] == self.category:
                positive.append(x)
                positive_true.append(self.full_train.target[i])
            else:
                negative.append(x)
                negative_true.append(self.full_train.target[i])
        
        self.positive = np.asarray(positive)
        self.negative = np.asarray(negative)
        self.negative_true = np.asarray(negative_true)
        self.positive_true = np.asarray(positive_true)

    def make_pu(self,gamma):
        P_size = int(self.positive.shape[0]*gamma)
        self.P = self.positive[:P_size]
        unlabeled_positive = self.positive[P_size:]
        self.U = np.concatenate((self.negative,unlabeled_positive))
        self.U_labels = np.zeros(self.U.shape[0])
        self.U_labels[self.negative.shape[0]:] = np.ones(unlabeled_positive.shape[0])
        self.true_labels = np.concatenate((self.negative_true,
                            self.positive_true[P_size:],self.positive_true[:P_size]))

    def make_rn(self,rn_indices):
        self.RN = self.U[rn_indices]
        q_indices = np.arange(len(self.U_labels))
        q_indices[rn_indices] = -1
        q_indices = [i for i in q_indices if i >= 0]
        self.Q = self.U[q_indices]

    def add_scores_dict(self,score,test_name):
        if test_name not in self.scores_dict:
            self.scores_dict[test_name] = []
        self.scores_dict[test_name].append(score)

    def test_data(self):
        test_set = fetch_20newsgroups(subset='test',
                        remove=('headers', 'footers', 'quotes'),
                        categories=self.cats)
        self.test_X = self.vectorizer.transform(test_set.data)
        self.test_X = self.test_X.toarray()
        #self.test_vector_length = self.test_X.shape[1]
        tf_labels = np.asarray(test_set.target) == self.category
        self.test_labels = tf_labels


def save_dataset(dataset,filename):
    with open(filename, 'wb') as dataset_file:
        pickle.dump(dataset, dataset_file)

def load_dataset(filename):
    with open(filename, 'rb') as dataset_file:
        dataset = pickle.load(dataset_file)
    return dataset



if __name__ == '__main__':
    dataset = Dataset()
    print("neg: ",dataset.negative.shape)
    print("pos: ",dataset.positive.shape)
    print("P: ",dataset.P.shape)
    print("U: ", dataset.U.shape)
    print("U_labels: ",dataset.U_labels.shape)
    print(dataset.test_labels)
    #save_dataset(dataset,'testing.data')

