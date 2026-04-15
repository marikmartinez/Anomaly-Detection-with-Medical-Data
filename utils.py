from traceback import print_tb

import sklearn
import numpy as np
import scipy
import pandas as pd
import streamlit as st

import data_utils


def getKthNearestNeighborsDistance(data, k):
    # TODO: make this more readable bc this looks fucked lol
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    print("K", k)

    # each list corresponds to each data point
    # len(distances) is the number of rows in the dataframe
    # each item is a list of distances to the 0-kth nearest neighbors. There are k distances in each of these lists.
    print("DISTANCES", len(distances)) 
    print("DISTANCES.SHAPE", distances.shape) 

    # To get the distance of the kth nearest neighbor, we should get the last item of each of these lists
    # Get the distance to kth nearest neighbor of each datapoint
    kNearestNeighborDistances = np.sort(distances[:, -1])
    print("KNNDISTANCES.SHAPE", kNearestNeighborDistances.shape) 
    return kNearestNeighborDistances


def generateWCSSValues(data, k_range):
    wcss_list = []
    (xmin, xmax) = k_range

    for k_val in range(xmin, xmax):
        kmeans = sklearn.cluster.KMeans(n_clusters=k_val)
        kmeans.fit(data)

        # Inertia is another name for Total Winthin Cluster Sum of squares

        wcss_list.append(kmeans.inertia_)

    return wcss_list


def getGMMClusterAssignments(data, num_components, covariance_type):
    gmm = sklearn.mixture.GaussianMixture(n_components=num_components, covariance_type=covariance_type)
    gmm_clusters = gmm.fit_predict(data)
    # TODO: ADD OUTLIERS

    return gmm_clusters


# Gets both the outliers (cluster assignment -1) and cluster assignments
def getDBSCANClusterAssignments(data, epsilon, min_samples):
    dbscan = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan_clusters = dbscan.fit_predict(data)

    return dbscan_clusters

#def getClusterAssignmentsGaussian(data, epsilon, min_samples):


# Gets both the outliers (cluster assignment -1) and cluster assignments
def getKMeansClusterAssignments(data, k, outlierThreshPercentile=95):
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(data)

    kmeans_clusters = kmeans.predict(data)
    print("KMEANS CLUSTERS UNIQUE", np.unique(kmeans_clusters))

    distancesToCentroids = getDistancesToCentroids(data, kmeans_clusters, kmeans.cluster_centers_)
    print("DISTANCES SHAPE", distancesToCentroids.shape)
    print("MIN DISTANCE", np.min(distancesToCentroids))
    print("MAX DISTANCE", np.max(distancesToCentroids))
    print("MEAN DISTANCE", np.mean(distancesToCentroids))

    #print("DISTANCES TO CENTROIDS", list(np.sort(distancesToCentroids)))
    outlierThresh = np.percentile(np.sort(distancesToCentroids), outlierThreshPercentile)
    print("OUTLIER THRESH", outlierThresh)
    print("UNIQUE CLUSTERS", np.unique(kmeans_clusters))

    new_kmeans_clusters = []

    for (distance, cluster) in zip(distancesToCentroids, kmeans_clusters):
        if distance < outlierThresh:
            new_kmeans_clusters.append(cluster)
        else:
            new_kmeans_clusters.append(-1) # -1 for outlier
            #print("HIT! DISTANCE", distance, "OUTLIER THRESH", outlierThresh)

    return new_kmeans_clusters

# Used for kmeans
def getDistancesToCentroids(data, clusters, centroids):
    print("data SHAPE", data.shape)

    # Gets the corresponding centroid for each data point (masking?)
    assignedCentroids = np.array(centroids)[clusters]

    # Get euclidean dist.
    distancesToCentroids = np.linalg.norm(data.values - assignedCentroids, axis=1)

    return distancesToCentroids


#def getClusterAssignments(data, num_gaussians, covariance_type):

# Use the output of this to plot & determine best model
# Source: sklearn Gaussian Model Selection page
# returns a df of the results
@st.cache_data
def GMMComparison(data):
    parameter_grid = {
        "n_components": range(1, 10),
        "covariance_type": ["spherical", "tied", "diag", "full"]
    }

    grid_search = sklearn.model_selection.GridSearchCV(sklearn.mixture.GaussianMixture(), param_grid=parameter_grid, scoring=gmm_bic_score)
    grid_search.fit(data)

    result_df = pd.DataFrame(grid_search.cv_results_)[
        [
            "param_n_components",
            "param_covariance_type",
            "mean_test_score",
        ]
    ]

    result_df["mean_test_score"] = -result_df["mean_test_score"]

    result_df = result_df.rename(
        columns={
            "param_n_components": "Number of Gaussian Components",
            "param_covariance_type": "Covariance Type",
            "mean_test_score": "BIC Score",
        }
    )

    return result_df

# Got from the source
def gmm_bic_score(estimator, X):
    return -estimator.bic(X)


def getPCADf(data):
    pca = sklearn.decomposition.PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    # TODO: prob add more stuff
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    return pca_df



