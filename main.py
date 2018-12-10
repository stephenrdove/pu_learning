import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import argparse
import plots

from create_data import Dataset,load_dataset
from rn_models import nb_clf,rns_clf,cluster_clf
from classifiers import create_svm,create_label_prop,test_clf

def create_and_test(dataset,clf_func,rn_test):
    clf = clf_func(dataset)
    return test_clf(clf,dataset,rn_test) 

def main(gamma=0.5,score_dict={}):
    cats = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    data = Dataset(cats=cats,gamma=gamma)
    data.scores_dict = score_dict

    if 'all' in args.rn:
        args.rn = 'nb cluster rns'

    if 'nb' in args.rn:
        nb_clf(data)
        y_svm = create_and_test(data,create_svm,'NB + SVM')
        y_lab = create_and_test(data,create_label_prop,'NB + Label Prop')
        if args.plot:
            plots.create_plots(np.concatenate((data.U,data.P)),y_svm,args.plot,'nb_svm')
            plots.create_plots(np.concatenate((data.U,data.P)),y_lab,args.plot,'nb_prop')

    
    if 'cluster' in args.rn: 
        cluster_clf(data,20,threshold=0.9)
        y_svm = create_and_test(data,create_svm,'Cluster + SVM')
        y_lab = create_and_test(data,create_label_prop,'Cluster + Label Prop')
        if args.plot:
            plots.create_plots(np.concatenate((data.U,data.P)),y_svm,args.plot,'clust_svm')
            plots.create_plots(np.concatenate((data.U,data.P)),y_lab,args.plot,'clust_prop')

    if 'rns' in args.rn:
        for i in 800*np.arange(1,2): #100*np.arange(1,10):
            rns_clf(data,i)
            create_and_test(data,create_svm,'RNS + SVM')
            create_and_test(data,create_label_prop,'RNS + Label Prop')
            if args.plot:
                plots.create_plots(np.concatenate((data.U,data.P)),y_svm,args.plot,'rns_svm')
                plots.create_plots(np.concatenate((data.U,data.P)),y_lab,args.plot,'rns_prop')
    
    return data
    
def gamma_plots():
    gammas = [0.2,0.4,0.5,0.6,0.8]
    score_dict = {}
    for gamma in gammas:
        data = main(gamma=gamma, score_dict=score_dict)
        score_dict = data.scores_dict
        print(score_dict)
    
    plots.basic_plot(score_dict,gammas,'all')

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data file", default="data")
    parser.add_argument("--plot", help="create plots?")
    parser.add_argument("--rn", type=str, help="nb, rns, cluster, or all", default="all")
    parser.add_argument("--restore", help="filepath to restore")
    args = parser.parse_args()
    
    if 'gamma' in args.plot:
        gamma_plots()
    else:
        main()
    
