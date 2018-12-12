import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold

def create_plots(X,y,arg_plots,fname_ext,data=None):
    if 'tsne' in arg_plots:
        tsne_plot(X,y,fname_ext)
    #if 'gamma' in arg_plots:
    #    basic_plot(data.scores_dict,[],fname_ext)


def tsne_plot(X,y,fname_ext):
    red = y == 0
    green = y == 1

    tsne = manifold.TSNE(n_components=2, init='random', perplexity=5)
    print('Running T-SNE')
    Y = tsne.fit_transform(X)
    print('Starting Plot...')
    plt.title("Perplexity=%d" % 5)
    plt.scatter(Y[red, 0], Y[red, 1], c="r")
    plt.scatter(Y[green, 0], Y[green, 1], c="g")
    plt.legend(('Negative','Positive'))

    fname = '../figures/tsne_' + fname_ext
    plt.savefig(fname=fname)

def basic_plot(scores_dict,x_axis,fname_ext):
    plt.title('RNS: F1 Score by Sample Size')
    legend_names = []
    # scores_dict = {legend name: array of scores}
    for score in scores_dict:
        legend_names.append(score)
        plt.plot(x_axis,scores_dict[score])

    plt.xlabel('Fraction of U')
    plt.ylabel('F1 Score')
    plt.legend(legend_names)

    fname = '../figures/threshold_' + fname_ext
    plt.savefig(fname=fname)

def box_plot(scores_dict,fname_ext):
    plt.title('Clusters: F1 Scores')
    legend_names = []
    scores = []
    # scores_dict = {legend name: array of scores}
    for score in scores_dict:
        legend_names.append(score)
        scores.append(scores_dict[score])


    plt.xlabel('Cluster Method')
    plt.ylabel('F1 Score')
    plt.boxplot(scores,labels=legend_names)

    fname = '../figures/box2_' + fname_ext
    plt.savefig(fname=fname)

def latex_table(scores_dict):

    print('\\begin{tabular}{|c|c|c|c|}')
    print('\t\\hline')
    print('\tModel & F1 Score & Precision & Recall \\\\')
    print('\t\\hline')
    for score in scores_dict:
        print('\t',score,'&',np.round(scores_dict[score][0][0],2)
                ,'&',np.round(scores_dict[score][0][1],2)
                ,'&',np.round(scores_dict[score][0][2],2),'\\\\')
        print('\t\\hline')
    print('\\end{tabular}')

