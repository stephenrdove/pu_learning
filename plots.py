import numpy as numpy
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
    plt.title('F1 Score by Gamma Value')
    legend_names = []
    # scores_dict = {legend name: array of scores}
    for score in scores_dict:
        legend_names.append(score)
        plt.plot(x_axis,scores_dict[score])

    plt.xlabel('Gamma Value')
    plt.ylabel('F1 Score')
    plt.legend(legend_names)

    fname = '../figures/f1bygamma_' + fname_ext
    plt.savefig(fname=fname)


