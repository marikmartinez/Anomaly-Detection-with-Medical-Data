import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import utils
import numpy as np

def plotGMMModelComparison(data):
    plt.cla()
    plt.clf()
    comparison_df = utils.GMMComparison(data)

    sns.catplot(
         data=comparison_df,
         kind="bar",
         x="Number of Gaussian Components",
         y="BIC Score",
         hue="Covariance Type",
    )
    plt.savefig("GMM_Model_Comparison_Plot.png")
    fig, ax = plt.subplots()
    return fig

def makeElbowPlot(yPoints, method, xlims=None):
    plt.cla()
    plt.clf()
    xPoints = list(range(1, len(yPoints) + 1))
    plt.plot(xPoints, yPoints)

    labelDict = {"dbscan": ("Data Point", "Distance to Kth Nearest Neighbor"), "kmeans": ("K-values", "Within-Cluster Sum of Squares")}
    xlabel, ylabel = labelDict[method]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # TODO: make this graph interactive or have something that figures out where this elbow window is
    if xlims is not None:
        plt.xlim(xlims)

    plt.savefig(f"{method}_elbow_plot.png")
    fig, ax = plt.subplots()
    return fig

def plotPCAWithClusters(data, clusterAssignments, method):
    plt.cla()
    plt.clf()

    print("DATA SHAPE", data.shape)
    print("DATA TAIL", data.tail())
    pca = sklearn.decomposition.PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])


    print("SINGULAR VALUES", pca.singular_values_)
    print("PCA DF", pca_df)
    print("EXPLAINED VARIANCE", pca.explained_variance_)

    scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c = clusterAssignments)
    plt.legend(*scatter.legend_elements(), title="Clusters")

    # Getting the principal component vectors to plot
    explained_variance = pca.explained_variance_ratio_

    loadings = pca.components_.T * np.sqrt(explained_variance)
    # Calculate scaling factor based on data range
    x_min, x_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    y_min, y_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    # Scale factor - adjust as needed
    scale = 1.5

    for i in range(len(data.columns)):
        vector_x = loadings[i, 0] * (x_max - x_min) * scale
        vector_y = loadings[i, 1] * (y_max - y_min) * scale
        # Add the name of the variable near the arrow
        plt.annotate(data.columns[i],  # variable name
                     (vector_x,
                      vector_y),
                     color='red')

        # Add an arrow representing the variable on the new axis
        plt.arrow(0, 0,
                  vector_x,
                  vector_y,
                  color='black',
                  alpha=0.7,
                  width=0.01,
                  )

    #plt.show()
    plt.savefig(f"{method}_pca.png")
    fig, ax = plt.subplots()
    return fig

    





