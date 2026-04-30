import streamlit as st
import pandas as pd
import numpy as np

import data_utils
import plot
import utils


# More readable way of "toggling" a button
def toggle_button(button_name, boolean):
    st.session_state[button_name] = boolean

# Have to reset everything (used when dataset is updated)
# Essential to having everything not be updated each time something is changed
# Only updated when things that would actually change the way the graph looks is changed (like when the dataset
# is updated
# TODO: rewrite this this is a mess
def reset(soft_reset=False):

    # If they don't exist yet, initialize them to False otherwise some cases where I check this will crash
    if "gmm_set_params" not in st.session_state:
        toggle_button("gmm_set_params", False)

    if "dbscan_set_params" not in st.session_state:
        toggle_button("dbscan_set_params", False)

    if "kmeans_set_params" not in st.session_state:
        toggle_button("kmeans_set_params", False)

    if not soft_reset:
        # Reset all the variables
        # These ones in particular are used in like "if var is not None" if statements
        st.session_state.preprocessed_data = None
        st.session_state.cluster_assignments = None
        st.session_state.target = None
        st.session_state.raw_data_selected = None
        toggle_button("refresh_helper_plots", True)

        # Reset the set_params bool (all the plots need to be remade from scratch bc new cols were defined)
        # Shouldn't "reuse" the params from the other runs when there were different helper plots
        toggle_button("gmm_set_params", False)
        toggle_button("dbscan_set_params", False)
        toggle_button("kmeans_set_params", False)
    else:
        # Don't
        toggle_button("refresh_helper_plots", False)








reset(soft_reset=True)
# SIDEBAR ---------------------------------------------------------------------
with st.sidebar.expander('Data Selection', expanded=True):

    # Determine the dataset using st.selectbox
    dataset = st.selectbox(
        "Dataset:",
        ("Heart Dataset", "Vitals Dataset"),
        index=0
    )

    if st.button("Confirm Dataset"):
        reset()

        st.session_state.dataset = dataset
        if st.session_state.dataset == "Heart Dataset":
            st.session_state.data_processing_func = data_utils.load_heart_data
        elif st.session_state.dataset == "Vitals Dataset":
            st.session_state.data_processing_func = data_utils.load_clinical_data

        raw_data_full = st.session_state.data_processing_func(preprocess=False)

        st.session_state.raw_data_full = raw_data_full

    if st.session_state.get("dataset", None) is not None:
        st.session_state.selected_columns = list(st.multiselect(
            "Columns:",
            list(st.session_state.raw_data_full.columns),
            default=list(st.session_state.raw_data_full.columns)[:3],
        ))


        if st.toggle("Enable Categorical Class Filtering", value=False, on_change=toggle_button, args=["cat_class_filter", True]):
            print("Categorical Class Filtering enabled")

            if st.session_state.dataset == "Heart Dataset":
                categorical_cols = ["cp", "restecg", "thal", "slope", "ca"]

            if st.session_state.dataset == "Vitals Dataset":
                categorical_cols = ["cormack", "airway", "iv1", "preop_ecg", "department", "dx", "optype", "opname"]

            categorical_excluding_selected_cols = list(set(categorical_cols) - set(st.session_state.selected_columns))

            st.session_state.filter_col = st.selectbox(
                "Column To Filter With:",
                categorical_excluding_selected_cols
            )
            print("FILTER COL", st.session_state.filter_col)

            st.session_state.categories_to_keep = list(st.multiselect(
                "Categories to Keep:",
                st.session_state.raw_data_full[st.session_state.filter_col].unique()
                #st.session_state.raw_data_full[st.session_state.filter_col].astype(str).str.strip().str.lower().unique()
            ))
        else:
            toggle_button("cat_class_filter", False)
            st.session_state.filter_col = None
            st.session_state.categories_to_keep = None

        print("CAT_CLASS_FILTER", st.session_state["cat_class_filter"])

        if st.button("Apply Columns & Filtering"):
            reset()

            if st.session_state.get("cat_class_filter", None) is not None and st.session_state.cat_class_filter is True:
                print("Applying Categorical Class Filtering!!!!!")
                st.session_state.preprocessed_data =  st.session_state.data_processing_func(
                    selected_columns=tuple(st.session_state.selected_columns),
                    filterTuple = (st.session_state.filter_col, st.session_state.categories_to_keep), preprocess = True)


            else:
                print("NOT SUPPOSED TO BE HERE!!!!!")
                st.session_state.preprocessed_data =  st.session_state.data_processing_func(
                    selected_columns=tuple(st.session_state.selected_columns))

            print("Raw DATA FULL IDXS -----------------------------------", print(list(st.session_state.raw_data_full.index)))
            st.session_state.raw_data_selected = st.session_state.raw_data_full.loc[
                st.session_state.preprocessed_data.index]
            print("DEBUG DATA LEN", len(st.session_state.preprocessed_data))

            print("LEFTOVER IDXSSSSSSSSSSSSS", list(st.session_state.preprocessed_data.index))

            # if len(st.session_state.preprocessed_data.columns) > 10:
            #     if len(st.session_state.preprocessed_data.columns) < 1000:
            #         st.warning("")
            #
            # if len(st.session_state.preprocessed_data.columns) < 1000:
            #     st.warning("")

if "raw_data_selected" in st.session_state:
    with st.sidebar.expander('Data Information', expanded=True):

        # print("RIGHT BEFORE PCA NUM COLS", len(st.session_state.preprocessed_data.columns))
        #
        # with st.expander('Summary Statistics', expanded=True):
        #     st.table(st.session_state.raw_data_selected.describe())

        if st.session_state.preprocessed_data is not None:
            with st.expander('PCA Projection Colored by Variable', expanded=True):
                #combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.raw_data_full["target"]], axis=1)

                datapoint_idxs = st.session_state.preprocessed_data.index
                if st.session_state.dataset == "Heart Dataset":
                    color_col = st.selectbox("Variable to Color Points", list(st.session_state.raw_data_full.columns), index=0)
                    if st.session_state.refresh_helper_plots or st.session_state.last_color != color_col:
                        plot.plotPCAWithColors(st.session_state.preprocessed_data, st.session_state.raw_data_selected[color_col], "sidebar")
                        st.session_state.last_color = color_col

                if st.session_state.dataset == "Vitals Dataset":
                    # Can't have the full range of cols like in heart dataset bc there are so many more columns
                    # in vitals that doing dropna() drops a bunch of rows unlike with the heart dataset
                    color_col = st.selectbox("Variable to Color Points", list(st.session_state.preprocessed_data.columns), index=0)

                    if st.session_state.refresh_helper_plots or st.session_state.last_color != color_col:
                        plot.plotPCAWithColors(st.session_state.preprocessed_data, st.session_state.raw_data_selected[color_col], "sidebar")
                        st.session_state.last_color = color_col

                st.image("sidebar_pca.png")


col1, col2, col3 = st.columns(3)

# KMEANS ---------------------------------------------------------------------
with col1:
    st.header("K-Means")


    with st.expander("K-Means Params", expanded=True):
        with st.form("kmeans_params"):
            k = st.slider("Number of components", 1, 10, 2)
            outlierThreshPercentile = st.slider("Outlier threshold percentile", 0, 100, 95)

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["kmeans_set_params", True])

        if submitted:
            toggle_button("kmeans_set_params", True)

        if st.session_state.get("preprocessed_data", None) is not None:
            with st.expander("Helper Plots", expanded=True):
                if st.session_state.refresh_helper_plots:
                    WCSS_values = utils.generateWCSSValues(st.session_state.preprocessed_data, (1, 10))

                    # Determine a good value for k
                    plot.makeElbowPlot(WCSS_values, "kmeans")
                st.image("kmeans_elbow_plot.png")

    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.kmeans_set_params is True and st.session_state.get("preprocessed_data", None) is not None:

        #pca_df = utils.getPCADf(st.session_state.preprocessed_data)

        if submitted:
            kmeans_clusters = utils.getKMeansClusterAssignments(st.session_state.preprocessed_data, k,
                                                                outlierThreshPercentile=outlierThreshPercentile)
            # This is probably not super efficient but it's probably fine
            curr_cluster_assignments = st.session_state.get("cluster_assignments", None)
            if curr_cluster_assignments is None:
                curr_cluster_assignments = pd.DataFrame()
            curr_cluster_assignments["kmeans_clusters"] = kmeans_clusters
            st.session_state.cluster_assignments = curr_cluster_assignments
            plot.plotPCAWithColors(st.session_state.preprocessed_data, kmeans_clusters, "kmeans")

        st.image("kmeans_pca.png")

    if st.session_state.get("cluster_assignments", None) is not None and "kmeans_clusters" in st.session_state.cluster_assignments:

        with st.expander("Cluster Summary Stats", expanded=True):
            col = st.selectbox(
                "Column",
                (list(st.session_state.raw_data_selected.columns)),
                index=0,
                key="cluster_summary_stats_kmeans"
            )
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
            st.table(combined_df.groupby("kmeans_clusters")[col].describe())

        # if st.session_state.dataset == "Heart Dataset":
        #     with st.expander("Outliers vs Heart Disease Patients", expanded=True):
        #
        #         clusters = np.array(st.session_state.cluster_assignments["kmeans_clusters"])
        #
        #         anomalies = np.where(clusters == -1, 1, 0)
        #
        #         heart_disease = st.session_state.raw_data_full["target"]
        #
        #         if st.session_state.refresh_helper_plots:
        #             plot.plotConfusionMatrix(heart_disease, anomalies)
        #         st.image("confusion_matrix.png")
        #


# DBSCAN ---------------------------------------------------------------------

with col2:
    st.header("DBSCAN")
    with st.expander("DBSCAN Params", expanded=True):

        if "dbscan_set_params" not in st.session_state:
            st.session_state.dbscan_set_params = False


        with st.form("dbscan_params"):
            epsilon = st.number_input("Epsilon", min_value=0.0, value=1.0)

            if "preprocessed_data" in st.session_state and st.session_state.preprocessed_data is not None:

                min_samples = st.number_input("Min Samples", min_value=3, value=2 * len(st.session_state.preprocessed_data.columns))
                st.session_state.dbscan_min_samples = min_samples

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["dbscan_set_params", True])

            if submitted:
                toggle_button("dbscan_set_params", True)

        if st.session_state.get("preprocessed_data", None) is not None and "dbscan_min_samples" in st.session_state :
            with st.expander("Helper Plots", expanded=True):
                if st.session_state.refresh_helper_plots:
                    kthNearestNeighborDistance = utils.getKthNearestNeighborsDistance(st.session_state.preprocessed_data, st.session_state.dbscan_min_samples)
                    plot.makeElbowPlot(kthNearestNeighborDistance, "dbscan")
                st.image("dbscan_elbow_plot.png")


    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.dbscan_set_params is True and st.session_state.get("preprocessed_data", None) is not None:
        if submitted:
            dbscan_clusters = utils.getDBSCANClusterAssignments(st.session_state.preprocessed_data, epsilon,
                                                                min_samples)
            curr_cluster_assignments = st.session_state.get("cluster_assignments", None)

            if curr_cluster_assignments is None:
                curr_cluster_assignments = pd.DataFrame()

            # This is probably not the most efficient way but it works
            curr_cluster_assignments["dbscan_clusters"] = dbscan_clusters

            st.session_state.cluster_assignments = curr_cluster_assignments

            plot.plotPCAWithColors(st.session_state.preprocessed_data, dbscan_clusters, "dbscan")
        st.image("dbscan_pca.png")

    if st.session_state.get("cluster_assignments", None) is not None and "dbscan_clusters" in st.session_state.cluster_assignments:


        with st.expander("Cluster Summary Stats", expanded=True):
            col = st.selectbox(
                "Column",
                (list(st.session_state.raw_data_selected.columns)),
                index=0,
                key="cluster_summary_stats_dbscan"
            )
            combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.cluster_assignments], axis=1)
            st.table(combined_df.groupby("dbscan_clusters")[col].describe())

        # if st.session_state.dataset == "Heart Dataset" and st.session_state.dbscan_set_params:
        #     with st.expander("Outliers vs Heart Disease Patients", expanded=True):
        #         clusters = np.array(st.session_state.cluster_assignments["dbscan_clusters"])
        #
        #         anomalies = np.where(clusters == -1, 1, 0)
        #
        #         heart_disease = st.session_state.target
        #
        #         if st.session_state.refresh_helper_plots:
        #             plot.plotConfusionMatrix(heart_disease, anomalies)
        #         st.image("confusion_matrix.png")

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
            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["gmm_set_params", True])

            if submitted:
                toggle_button("gmm_set_params", True)

        if st.session_state.get("preprocessed_data", None) is not None:
            with st.expander("Helper Plots", expanded=True):
                if st.session_state.refresh_helper_plots:
                    plot.plotGMMModelComparison(st.session_state.preprocessed_data)
                st.image("gmm_model_comparison_plot.png")



    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state.gmm_set_params is True and st.session_state.get("preprocessed_data", None) is not None:

        if submitted:
            gmm_clusters = utils.getGMMClusterAssignments(st.session_state.preprocessed_data, num_components,
                                                          covariance_type,
                                                          outlierThreshPercentile=outlierThreshPercentile)

            curr_cluster_assignments = st.session_state.get("cluster_assignments", None)
            if curr_cluster_assignments is None:
                curr_cluster_assignments = pd.DataFrame()
            curr_cluster_assignments["gmm_clusters"] = gmm_clusters
            st.session_state.cluster_assignments = curr_cluster_assignments
            plot.plotPCAWithColors(st.session_state.preprocessed_data, gmm_clusters, "gmm")

        st.image("gmm_pca.png")

    if st.session_state.get("cluster_assignments", None) is not None and "gmm_clusters" in st.session_state.cluster_assignments:
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

        # if st.session_state.dataset == "Heart Dataset" and st.session_state.gmm_set_params:
        #     with st.expander("Outliers vs Heart Disease Patients", expanded=True):
        #         clusters = np.array(st.session_state.cluster_assignments["gmm_clusters"])
        #
        #         anomalies = np.where(clusters == -1, 1, 0)
        #
        #         heart_disease = st.session_state.target
        #
        #         if st.session_state.refresh_helper_plots:
        #             plot.plotConfusionMatrix(heart_disease, anomalies)
        #         st.image("confusion_matrix.png")




print("REFRESH PLOTS BEFORE", st.session_state.refresh_helper_plots)
toggle_button("refresh_helper_plots", False)
print("REFRESH PLOTS AFTER", st.session_state.refresh_helper_plots)
