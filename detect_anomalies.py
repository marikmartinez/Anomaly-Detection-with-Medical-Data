
import data_utils
import numpy as np
import utils
import plot


if __name__=="__main__":
    #raw_data = data_utils.load_clinical_data()
    raw_data = data_utils.load_heart_data(preprocess=False)

    heart_disease_list = raw_data[["target"]]
    print("NUM ROWS", len(raw_data))

    print("COLUMNS", raw_data.columns)
    preprocessed_data = data_utils.load_heart_data()
    # TODO: look into this issue
    # Why do I have to do this again??? I alr did this in load_clinical_preprocessed_data but it says it has nas
    preprocessed_data=preprocessed_data.dropna()
    print(list(preprocessed_data.columns))

    # DBSCAN -----------------------------------------------------------------------------------------

    min_samples = 2 * len(preprocessed_data.columns)  # Rule of thumb: min_samples = 2 * num_dims
    kthNearestNeighborDistance = utils.getKthNearestNeighborsDistance(preprocessed_data, min_samples)
    plot.makeElbowPlot(kthNearestNeighborDistance, "dbscan")

    epsilon = 0.8 #TODO: do elbow plot to get a number for this
    #epsilon = 3.5 #TODO: do elbow plot to get a number for this

    print("MIN SAMPLES", min_samples)


    dbscan_clusters = utils.getDBSCANClusterAssignments(preprocessed_data, epsilon, min_samples)

    # Plotting the pca with the cluster colors
    plot.plotPCAWithClusters(preprocessed_data, dbscan_clusters, "dbscan")

    preprocessed_data["dbscan_clusters"] = dbscan_clusters
    print("NUM CLUSTERS", len(np.unique(dbscan_clusters)))
    print("CLUSTERS", np.unique(dbscan_clusters))

    # K MEANS -------------------------------------------------

    WCSS_values= utils.generateWCSSValues(preprocessed_data, (1, 10))

    # Determine a good value for k
    plot.makeElbowPlot(WCSS_values, "kmeans")

    k=3

    kmeans_clusters = utils.getKMeansClusterAssignments(preprocessed_data, k)

    # TODO: maybe pass in categorical values (not onehot) so that we can color by category
    plot.plotPCAWithClusters(preprocessed_data, kmeans_clusters, "kmeans")

    # GMM --------------------------------------------
    # TODO: Fix this!!!!!!!
    plot.plotGMMModelComparison(preprocessed_data)


    gmm_clusters = utils.getGMMClusterAssignments(preprocessed_data, 4, "diag")
    plot.plotPCAWithClusters(preprocessed_data, gmm_clusters, "gmm")







