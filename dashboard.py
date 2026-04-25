import streamlit as st
import pandas as pd
import numpy as np

import data_utils
import plot
import utils

def toggle_button(button_name):
    st.session_state[button_name] = True

# SIDEBAR ---------------------------------------------------------------------
with st.sidebar.expander('Data Selection', expanded=True):
    # Using object notation
    dataset = st.selectbox(
        "Dataset:",
        ("Heart Dataset", "Vitals Dataset"),
        index=0
    )

    if dataset == "Heart Dataset":
        raw_data_full = data_utils.load_heart_data(preprocess=False)
    elif dataset == "Vitals Dataset":
        raw_data_full = data_utils.load_clinical_data(preprocess=False)

    st.session_state["raw_data_full"] = raw_data_full

    st.session_state.selected_columns = tuple(st.multiselect(
        "Columns:",
        list(raw_data_full.columns),
        default=list(raw_data_full.columns)[:3],
    ))


    if st.button("Apply Columns"):
        st.session_state.preprocessed_data =  data_utils.load_heart_data(
            selected_columns=st.session_state.selected_columns)
        st.session_state.raw_data_selected = data_utils.load_heart_data(
            selected_columns=tuple(list(st.session_state.selected_columns) + ["target"]), preprocess=False)
        st.session_state.target = st.session_state.raw_data_selected["target"]
        st.session_state.raw_data_selected.drop(columns=["target"], inplace=True)



if "raw_data_selected" in st.session_state:
    with st.sidebar.expander('Data Information', expanded=True):
        with st.expander('The Dataset', expanded=False):
            if "cluster_assignments" not in st.session_state:
                st.table(st.session_state.raw_data_selected)
            else:
                combined_df=pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
                st.table(combined_df)


        with st.expander('Summary Statistics', expanded=True):
            st.table(st.session_state.raw_data_selected.describe())

        with st.expander('PCA Projection Colored by Variable', expanded=True):
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.raw_data_full["target"]], axis=1)
            color_col = st.selectbox("Variable to Color Points", list(combined_df.columns), index=0)

            plot.plotPCAWithColors(st.session_state.preprocessed_data, st.session_state.raw_data_full[color_col], "sidebar")
            st.image("sidebar_pca.png")


col1, col2, col3 = st.columns(3)

# KMEANS ---------------------------------------------------------------------
with col1:
    st.header("K-Means")

    if "kmeans_set_params" not in st.session_state:
        st.session_state.kmeans_set_params = False

    with st.expander("K-Means Params", expanded=True):
        with st.form("kmeans_params"):
            k = st.slider("Number of components", 1, 10, 2)
            outlierThreshPercentile = st.slider("Outlier threshold percentile", 0, 100, 95)


            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["kmeans_set_params"])

        if "preprocessed_data" in st.session_state:
            with st.expander("Helper Plots", expanded=True):
                WCSS_values = utils.generateWCSSValues(st.session_state.preprocessed_data, (1, 10))

                # Determine a good value for k
                plot.makeElbowPlot(WCSS_values, "kmeans")
                st.image("kmeans_elbow_plot.png")

    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.kmeans_set_params is True and "preprocessed_data" in st.session_state:
        kmeans_clusters = utils.getKMeansClusterAssignments(st.session_state.preprocessed_data, k, outlierThreshPercentile=outlierThreshPercentile)
        # This is probably not super efficient but it's probably fine
        curr_cluster_assignments = st.session_state.get("cluster_assignments", pd.DataFrame())
        curr_cluster_assignments["kmeans_clusters"] = kmeans_clusters
        st.session_state.cluster_assignments = curr_cluster_assignments

        #pca_df = utils.getPCADf(st.session_state.preprocessed_data)

        plot.plotPCAWithColors(st.session_state.preprocessed_data, kmeans_clusters, "kmeans")

        st.image("kmeans_pca.png")

    if "cluster_assignments" in st.session_state and "kmeans_clusters" in st.session_state.cluster_assignments:

        with st.expander("Cluster Summary Stats", expanded=True):
            col = st.selectbox(
                "Column",
                (list(st.session_state.raw_data_selected.columns)),
                index=0,
                key="cluster_summary_stats_kmeans"
            )
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
            st.table(combined_df.groupby("kmeans_clusters")[col].describe())


        with st.expander("Outliers vs Heart Disease Patients", expanded=True):

            clusters = np.array(st.session_state.cluster_assignments["kmeans_clusters"])

            anomalies = np.where(clusters == -1, 1, 0)

            heart_disease = st.session_state.target


            plot.plotConfusionMatrix(heart_disease, anomalies)
            st.image("confusion_matrix.png")



# DBSCAN ---------------------------------------------------------------------

with col2:
    st.header("DBSCAN")
    with st.expander("DBSCAN Params", expanded=True):

        if "dbscan_set_params" not in st.session_state:
            st.session_state.dbscan_set_params = False


        with st.form("dbscan_params"):
            epsilon = st.number_input("Epsilon", min_value=0.0, value=1.0)

            if "preprocessed_data" in st.session_state:

                min_samples = st.number_input("Min Samples", min_value=3, value=2 * len(st.session_state.preprocessed_data.columns))
                st.session_state.dbscan_min_samples = min_samples

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["dbscan_set_params"])

        if "preprocessed_data" in st.session_state and "dbscan_min_samples" in st.session_state:
            with st.expander("Helper Plots", expanded=True):
                kthNearestNeighborDistance = utils.getKthNearestNeighborsDistance(st.session_state.preprocessed_data, st.session_state.dbscan_min_samples)
                plot.makeElbowPlot(kthNearestNeighborDistance, "dbscan")
                st.image("dbscan_elbow_plot.png")


    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.dbscan_set_params is True and "preprocessed_data" in st.session_state:
        dbscan_clusters = utils.getDBSCANClusterAssignments(st.session_state.preprocessed_data, epsilon, min_samples)
        curr_cluster_assignments = st.session_state.get("cluster_assignments", pd.DataFrame())

        # This is probably not the most efficient way but it works
        curr_cluster_assignments["dbscan_clusters"] = dbscan_clusters

        plot.plotPCAWithColors(st.session_state.preprocessed_data, dbscan_clusters, "dbscan")

        st.image("dbscan_pca.png")

    if "cluster_assignments" in st.session_state and "dbscan_clusters" in st.session_state.cluster_assignments:

        with st.expander("Cluster Summary Stats", expanded=True):
            col = st.selectbox(
                "Column",
                (list(st.session_state.raw_data_selected.columns)),
                index=0,
                key="cluster_summary_stats_dbscan"
            )
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
            st.table(combined_df.groupby("dbscan_clusters")[col].describe())

        with st.expander("Outliers vs Heart Disease Patients", expanded=True):
            clusters = np.array(st.session_state.cluster_assignments["dbscan_clusters"])

            anomalies = np.where(clusters == -1, 1, 0)

            heart_disease = st.session_state.target

            plot.plotConfusionMatrix(heart_disease, anomalies)
            st.image("confusion_matrix.png")

# GMM ---------------------------------------------------------------------
with col3:
    st.header("GMM")

    with st.expander("GMM Params", expanded=True):

        if "gmm_set_params" not in st.session_state:
            st.session_state.gmm_set_params = False

        with st.form("gmm_params"):
            num_components = st.slider("Number of components", 1, 10, 2)
            covariance_type = st.selectbox(
                "Covariance type",
                ("full", "spherical", "tied", "diag"),
                index=0)

            outlierThreshPercentile = st.slider("Outlier threshold percentile", 0, 100, 5)
            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["gmm_set_params"])

        if "preprocessed_data" in st.session_state:
            with st.expander("Helper Plots", expanded=True):
                plot.plotGMMModelComparison(st.session_state.preprocessed_data)
                st.image("gmm_model_comparison_plot.png")



    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.gmm_set_params is True and "preprocessed_data" in st.session_state:
        gmm_clusters = utils.getGMMClusterAssignments(st.session_state.preprocessed_data, num_components, covariance_type, outlierThreshPercentile=outlierThreshPercentile)

        curr_cluster_assignments = st.session_state.get("cluster_assignments", pd.DataFrame())
        curr_cluster_assignments["gmm_clusters"] = gmm_clusters

        plot.plotPCAWithColors(st.session_state.preprocessed_data, gmm_clusters, "gmm")

        st.image("gmm_pca.png")

    if "cluster_assignments" in st.session_state and "gmm_clusters" in st.session_state.cluster_assignments:
        with st.expander("Cluster Summary Stats", expanded=True):
            col = st.selectbox(
                "Column",
                (list(st.session_state.raw_data_selected.columns)),
                index=0,
                key = "cluster_summary_stats_gmm"
            )
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
            # Group by gmm_clusters then select only data for this specific column (summary stats on this col only)
            st.table(combined_df.groupby("gmm_clusters")[col].describe())

        with st.expander("Outliers vs Heart Disease Patients", expanded=True):
            clusters = np.array(st.session_state.cluster_assignments["gmm_clusters"])

            anomalies = np.where(clusters == -1, 1, 0)

            heart_disease = st.session_state.target

            plot.plotConfusionMatrix(heart_disease, anomalies)
            st.image("confusion_matrix.png")



