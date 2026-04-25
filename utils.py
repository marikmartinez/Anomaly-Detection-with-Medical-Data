import sklearn
import numpy as np
import pandas as pd



# Used to make elbow plot for DBSCAN (finding epsilon)
def getKthNearestNeighborsDistance(data, k):
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # each list corresponds to each data point
    # len(distances) is the number of rows in the dataframe
    # each item is a list of distances to the 0-kth nearest neighbors. There are k distances in each of these lists.

    # To get the distance of the kth nearest neighbor, we should get the last item of each of these lists
    # Get the distance to kth nearest neighbor of each datapoint
    kNearestNeighborDistances = np.sort(distances[:, -1])
    return kNearestNeighborDistances


# Used to make elbow plot for kmeans (finding k)
def generateWCSSValues(data, k_range):
    wcss_list = []
    (xmin, xmax) = k_range

    for k_val in range(xmin, xmax):
        kmeans = sklearn.cluster.KMeans(n_clusters=k_val)
        kmeans.fit(data)

        # Inertia is another name for Total Within Cluster Sum of Squares
        wcss_list.append(kmeans.inertia_)

    return wcss_list

# Using chosen parameters, use gmm to cluster
def getGMMClusterAssignments(data, num_components, covariance_type, outlierThreshPercentile=5):
    # Cluster normally
    gmm = sklearn.mixture.GaussianMixture(n_components=num_components, covariance_type=covariance_type)
    gmm_clusters = gmm.fit_predict(data)

    # Use log probs and threshold it based on percentile to determine the outliers
    log_probs = gmm.score_samples(data)

    outlierThresh = np.percentile(log_probs, outlierThreshPercentile)

    # Find outliers and replace their cluster assignments with -1
    new_gmm_clusters = np.where(
        log_probs < outlierThresh,
        -1,
        gmm_clusters
    )

    return new_gmm_clusters


# Gets both the outliers (cluster assignment -1) and cluster assignments
def getDBSCANClusterAssignments(data, epsilon, min_samples):
    dbscan = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan_clusters = dbscan.fit_predict(data)

    return dbscan_clusters


# Gets both the outliers (cluster assignment -1) and cluster assignments
def getKMeansClusterAssignments(data, k, outlierThreshPercentile=95):
    # Cluster normally
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(data)

    kmeans_clusters = kmeans.predict(data)

    # Use distances to centroids to determine outlierThresh
    distancesToCentroids = getDistancesToCentroids(data, kmeans_clusters, kmeans.cluster_centers_)

    #print("DISTANCES TO CENTROIDS", list(np.sort(distancesToCentroids)))
    outlierThresh = np.percentile(np.sort(distancesToCentroids), outlierThreshPercentile)

    # Use outlier thresh to determine outliers
    new_kmeans_clusters = np.where(
        distancesToCentroids > outlierThresh,
        -1,
        kmeans_clusters
    )
    return new_kmeans_clusters

# Used for kmeans to determine outliers
def getDistancesToCentroids(data, clusters, centroids):
    print("data SHAPE", data.shape)

    # Gets the corresponding centroid for each data point (cool numpy stuff)
    assignedCentroids = np.array(centroids)[clusters]

    # Get euclidean dist.
    distancesToCentroids = np.linalg.norm(data.values - assignedCentroids, axis=1)

    return distancesToCentroids

# Use the output of this to plot & determine best model based on BIC (Bayesian Information Criterion)
# Lower BIC -> better
# Source: sklearn Gaussian Model Selection page
# returns a df of the results
def GMMComparison(data):
    # Different params to try
    parameter_grid = {
        "n_components": range(1, 10),
        "covariance_type": ["spherical", "tied", "diag", "full"] # try different covariance types to see which one performs best
    }

    # Try all these diff parameters on the data
    grid_search = sklearn.model_selection.GridSearchCV(sklearn.mixture.GaussianMixture(), param_grid=parameter_grid, scoring=gmm_bic_score)
    grid_search.fit(data)

    # put the results in a df so we can plot
    result_df = pd.DataFrame(grid_search.cv_results_)[
        [
            "param_n_components",
            "param_covariance_type",
            "mean_test_score",
        ]
    ]

    # Turn back to lower is better
    result_df["mean_test_score"] = -result_df["mean_test_score"]

    # Change names to be more understandable
    result_df = result_df.rename(
        columns={
            "param_n_components": "Number of Gaussian Components",
            "param_covariance_type": "Covariance Type",
            "mean_test_score": "BIC Score",
        }
    )

    return result_df

# Got from sklearn model selection page
# turning negative so bigger is better
# GridSearchCV tries to maximize
def gmm_bic_score(estimator, X):
    return -estimator.bic(X)


