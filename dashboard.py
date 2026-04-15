import streamlit as st

import data_utils
import plot
import utils

def toggle_button(button_name):
    st.session_state[button_name] = True


with st.sidebar.expander('Data Preparation', expanded=True):
    # Using object notation
    dataset = st.selectbox(
        "Dataset:",
        ("Heart Dataset", "Vitals Dataset"),
        index=0
    )

    if dataset == "Heart Dataset":
        raw_data = data_utils.load_heart_data(preprocess=False)
    elif dataset == "Vitals Dataset":
        raw_data = data_utils.load_clinical_data(preprocess=False)

    st.session_state["raw_data"] = raw_data

    st.session_state.selected_columns = tuple(st.multiselect(
        "Columns:",
        list(raw_data.columns),
        default=list(raw_data.columns)[:3],
    ))


    if st.button("Apply Columns"):
        st.session_state.preprocessed_data =  data_utils.load_heart_data(columns=st.session_state.selected_columns)


# # Using "with" notation
# with st.sidebar:
#         dataset = st.radio(
#                         "Dataset:",
#                                 ("Heart Dataset", "Vitals Dataset"),
#                         index=0
#                                     )

col1, col2, col3 = st.columns(3)

with col1:
    st.header("K-Means")

    if "kmeans_set_params" not in st.session_state:
        st.session_state.kmeans_set_params = False

    with st.expander("K-Means Params", expanded=True):
        with st.form("kmeans_params"):
            k = st.slider("Number of components", 1, 10, 2)
            outlierThreshPercentile = st.slider("Outlier threshold percentile", 0, 100, 95)

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["kmeans_set_params"])

    st.header("KMeans Clustering with PCA")
    if st.session_state.kmeans_set_params is True and "preprocessed_data" in st.session_state:
        print("PREPROCESSED DATA", st.session_state.preprocessed_data)
        kmeans_clusters = utils.getKMeansClusterAssignments(st.session_state.preprocessed_data, k, outlierThreshPercentile=outlierThreshPercentile)
        print("LEN PREPROCESSED DATA", len(st.session_state.preprocessed_data))
        print("LEN KMEANS CLUSTERS", len(kmeans_clusters))

        #pca_df = utils.getPCADf(st.session_state.preprocessed_data)

        plot.plotPCAWithClusters(st.session_state.raw_data, kmeans_clusters, "kmeans")

        st.image("kmeans_pca.png")








with col2:
    st.header("DBSCAN")
    with st.expander("DBSCAN Params", expanded=True):

        if "dbscan_set_params" not in st.session_state:
            st.session_state.dbscan_set_params = False

        with st.form("dbscan_params"):
            epsilon = st.number_input("Epsilon", min_value=0.0, value=1.0)
            min_samples = st.number_input("Min Samples", min_value=3, value=3)

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["dbscan_set_params"])


    st.header("DBSCAN Clustering with PCA")
    if st.session_state.dbscan_set_params is True and "preprocessed_data" in st.session_state:
        print("PREPROCESSED DATA", st.session_state.preprocessed_data)
        dbscan_clusters = utils.getDBSCANClusterAssignments(st.session_state.preprocessed_data, epsilon, min_samples)
        print("LEN PREPROCESSED DATA", len(st.session_state.preprocessed_data))

        # pca_df = utils.getPCADf(st.session_state.preprocessed_data)

        plot.plotPCAWithClusters(st.session_state.raw_data, dbscan_clusters, "dbscan")

        st.image("dbscan_pca.png")

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
                index=0
            )

            submitted = st.form_submit_button("Apply Params", on_click=toggle_button, args=["gmm_set_params"])



    st.header("GMM Clustering with PCA")
    if st.session_state.gmm_set_params is True and "preprocessed_data" in st.session_state:
        gmm_clusters = utils.getGMMClusterAssignments(st.session_state.preprocessed_data, num_components, covariance_type)

        # pca_df = utils.getPCADf(st.session_state.preprocessed_data)

        plot.plotPCAWithClusters(st.session_state.raw_data, gmm_clusters, "gmm")

        st.image("gmm_pca.png")



