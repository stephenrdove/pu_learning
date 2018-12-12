import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import argparse
import plots

from create_data import Dataset,load_dataset
from rn_models import nb_clf,rns_clf,cluster_clf
from classifiers import create_svm,create_label_prop,test_clf

"""
This function will create and test the final classifier given a dataset
that has the RN attribute

    :param dataset (Dataset): Dataset class instance
    :param clf_func (function): The function to use from classifiers
    :param rn_test (str): Name of the test for saving F1 Scores
    :param y (list): OPTIONAL - List of labels for Cluster Only classifier
"""
def create_and_test(dataset,clf_func,rn_test,y=None):
    if clf_func:
        clf = clf_func(dataset)
        return test_clf(clf,dataset,rn_test,y) 
    return test_clf(None,dataset,rn_test,y)


"""
This function is the main function that goes through 4 datasets and runs 
each test as specified by the --rn flag. 

    :param gamma (float): OPTIONAL - fraction of positive samples that are 
                            labeled as positive
    :param score_dicts (list): OPTIONAL - list of dicts to 
"""
def main(gamma=0.5,score_dicts=None):
    cats = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
    datasets = []
    for i in range(len(cats)):
        data = Dataset(cats=cats,gamma=gamma,category=i)
        if score_dicts:
            data.scores_dict = score_dicts[i]
        datasets.append(data)
    
    for data in datasets:
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
            #thresholds = [0.5,0.6,0.7,0.8,0.85,0.9,0.95]
            thresholds = [0.9]
            n_clusters = [20]
            for th in thresholds:
                for n_cluster in n_clusters:
                    cluster_clf(data,n_cluster,threshold=th)
                    y_svm = create_and_test(data,create_svm,'Cluster + SVM')
                    y_lab = create_and_test(data,create_label_prop,'Cluster + Label Prop')
                    clust_labels = cluster_clf(data,n_cluster,threshold=th,rn=False)
                    y_clust = create_and_test(data,None,'Cluster Only',clust_labels)
                    if args.plot:
                        plots.create_plots(np.concatenate((data.U,data.P)),y_svm,args.plot,'clust_svm')
                        plots.create_plots(np.concatenate((data.U,data.P)),y_lab,args.plot,'clust_prop')
                        plots.create_plots(np.concatenate((data.U,data.P)),y_clust,args.plot,'clust_only')
            
            #plots.box_plot(data.scores_dict,'clusters')
            #plots.basic_plot(data.scores_dict,thresholds,'clust')       

        if 'rns' in args.rn:
            U_size = data.U.shape[0]
            frac_U = [0.2]
            for i in frac_U: 
                rns_clf(data,int(np.ceil(i*U_size)))
                y_svm = create_and_test(data,create_svm,'RNS + SVM')
                y_lab = create_and_test(data,create_label_prop,'RNS + Label Prop')
                if args.plot:
                    plots.create_plots(np.concatenate((data.U,data.P)),y_svm,args.plot,'rns_svm')
                    plots.create_plots(np.concatenate((data.U,data.P)),y_lab,args.plot,'rns_prop')
        
            #plots.basic_plot(data.scores_dict,frac_U,'rns2')
        
    return datasets
    
def gamma_plots():
    gammas = [0.5]
    score_dicts = []
    tot_cats = 4
    for _ in range(tot_cats):
        score_dicts.append({})
    for gamma in gammas:
        datasets = main(gamma=gamma, score_dicts=score_dicts)
        score_dicts = []
        for data in datasets:
            score_dicts.append(data.scores_dict)
    
    final_scores = {}
    num_data = len(score_dicts)
    for scores in score_dicts:
        for score in scores:
            if score not in final_scores:
                final_scores[score] = 0
            final_scores[score] += np.asarray(scores[score])
    
    for score in final_scores:
        final_scores[score] = final_scores[score] / num_data
    
    plots.basic_plot(final_scores,gammas,'gammaplot')
    #plots.latex_table(final_scores)


        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", help="create plots?")
    parser.add_argument("--rn", type=str, help="nb, rns, cluster, or all", default="all")
    args = parser.parse_args()
    
    if args.plot and 'gamma' in args.plot:
        gamma_plots()
    else:
        main()
    
