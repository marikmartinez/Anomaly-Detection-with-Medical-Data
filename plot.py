import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
import utils
import numpy as np

# Plot gmm model comparison on cat plot
# Allows us to easily see lots of info on one graph
def plotGMMModelComparison(data):
    plt.cla()
    plt.clf()
    comparison_df = utils.GMMComparison(data)

    model_comparison = sns.catplot(
         data=comparison_df,
         kind="bar",
         x="Number of Gaussian Components",
         y="BIC Score",
         hue="Covariance Type",
    )

    # Ran into cases where some bars were super huge, making the rest of the graph unreadable
    # Setting ylims to avoid this
    model_comparison.set(ylim=(
        comparison_df["BIC Score"].quantile(0.05),
        comparison_df["BIC Score"].quantile(0.95)
    ))

    plt.savefig("gmm_model_comparison_plot.png")

def makeElbowPlot(yPoints, method, xlims=None):
    plt.cla()
    plt.clf()
    xPoints = list(range(1, len(yPoints) + 1))
    plt.plot(xPoints, yPoints)

    labelDict = {"dbscan": ("Data Point", "Distance to Kth Nearest Neighbor"), "kmeans": ("K-values", "Within-Cluster Sum of Squares")}
    xlabel, ylabel = labelDict[method]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlims is not None:
        plt.xlim(xlims)

    plt.savefig(f"{method}_elbow_plot.png")

# Using PCA to visualize
# Reduces the amt of dimensions to 2 so we can easily plot the 2 principal components
def plotPCAWithColors(data, colorAssignments, method):
    plt.cla()
    plt.clf()

    pca = sklearn.decomposition.PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # plotting the two principal components
    scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c = colorAssignments)
    plt.legend(*scatter.legend_elements(), title="Category")

    # Getting the principal component vectors to plot
    explained_variance = pca.explained_variance_ratio_

    plt.xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)")

    # Calculating how much each feature contributes to principal components
    loadings = pca.components_.T * np.sqrt(explained_variance)


    # Vectors are way too small, need to scale so they're proportional to the plot
    # Calculate scaling factor based on data range
    x_min, x_max = pca_df["PC1"].min(), pca_df["PC1"].max()
    y_min, y_max = pca_df["PC2"].min(), pca_df["PC2"].max()

    # Scale factor
    scale = 1

    # Plot the loading vectors
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

# Plot a confusion matrix (for comparing between outliers and heart disease patients)
def plotConfusionMatrix(true, pred):
    disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(true, pred)
    disp.figure_.savefig("confusion_matrix.png")





