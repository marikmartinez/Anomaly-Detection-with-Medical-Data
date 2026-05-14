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
        None,
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
    plt.title(f"{method} PCA")

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


def plotViolinPlot(data, x, y, color, plotName):
    plt.cla()
    plt.clf()

    violin_plot = sns.violinplot(data=data, x=x, y=y, hue=color, fill=True)

    plt.savefig(plotName)

def plotOutlierComparisonViolin(combined_df, feature_col, plotName):
    plt.cla()
    plt.clf()

    outlier_df = None

    for alg in ["kmeans", "dbscan", "gmm"]:
        if alg in combined_df.columns:
            alg_outliers = combined_df[combined_df[alg] == -1].copy()  # Fix 3: use .copy()
            alg_outliers["alg"] = alg

            cols_to_drop = [a for a in ["kmeans", "dbscan", "gmm"] if a in alg_outliers.columns]
            alg_outliers.drop(columns=cols_to_drop, inplace=True)

            if outlier_df is None:
                outlier_df = alg_outliers
            else:
                outlier_df = pd.concat([outlier_df, alg_outliers], ignore_index=True)

    assert outlier_df is not None

    # Build overall baseline row
    overall = combined_df[[feature_col]].copy()
    overall["alg"] = "overall"

    plot_df = pd.concat([overall, outlier_df[[feature_col, "alg"]]], ignore_index=True)

    # Color "overall" differently so it stands out
    alg_order = ["overall", "kmeans", "dbscan", "gmm"]

    sns.violinplot(data=plot_df, x="alg", y=feature_col, order=alg_order,
                   fill=False)

    plt.title(f"{feature_col} Distribution in Outliers By Algorithm (vs. Overall)")
    plt.xlabel("alg")
    plt.savefig(plotName)


def plotHistogram(data, feature_to_plot, plotName):
    plt.cla()
    plt.clf()

    plt.hist(data[feature_to_plot])
    plt.ylabel("Counts")
    plt.xlabel(feature_to_plot)
    plt.savefig(plotName)


def plotBarPlot(data, feature_to_plot, plotName):
    plt.cla()
    plt.clf()

    counts = data[feature_to_plot].value_counts().sort_index()
    plt.bar(counts.index, counts.values)
    plt.xticks(counts.index)  # force ticks at exactly 0 and 1
    plt.ylabel("Counts")
    plt.xlabel(feature_to_plot)
    plt.savefig(plotName)
