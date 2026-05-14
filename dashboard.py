from pydoc import render_doc

import streamlit as st
import pandas as pd
import numpy as np

import data_utils
import plot
import utils

from data_utils import HEART_COLS
from data_utils import VITAL_COLS

# More readable way of "toggling" a button
def toggle_button(button_name, boolean):
    st.session_state[button_name] = boolean

# Have to reset everything (used when dataset is updated)
# Essential to having everything not be updated each time something is changed
# Only updated when things that would actually change the way the graph looks is changed (like when the dataset
# is updated
# TODO: rewrite this this is a mess
def reset():
    if "initialized" not in st.session_state:
        st.session_state["dataset"] = None

    # If they don't exist yet, initialize them to False otherwise some cases where I check this will crash
    toggle_button("gmm_set_params", False)

    toggle_button("dbscan_set_params", False)

    # Reset the set_params bool (all the plots need to be remade from scratch bc new cols were defined)
    # Shouldn't "reuse" the params from the other runs when there were different helper plots
    for alg in ["kmeans", "dbscan", "gmm"]:
        st.session_state[alg] = {"params_changed":False,
                                 "user_params": None,
                                 "cluster_assgns": None,
                                 "helper_plot_param": None}

    # Reset all the variables
    # These ones in particular are used in like "if var is not None" if statements
    st.session_state.preprocessed_data = None
    st.session_state.target = None
    st.session_state.raw_data_selected = None
    st.session_state.last_color = None
    st.session_state.refresh_helper_plots = True
    st.session_state.refresh_summary_plots = False
    st.session_state.cluster_summary_all_cols = False
    st.session_state.summary_col = None


    # First initialization of vars is done
    st.session_state["initialized"] = True



def show_kmeans_param_selection():
    if st.session_state.dataset == "Heart Dataset":
        default_k = 3
    else:
        default_k = 5
    with st.expander("K-Means Params", expanded=True):
        with st.form("kmeans_params"):
            k = st.slider("Number of components", 1, 10, default_k)
            outlierThreshPercentile = st.slider("Outlier threshold percentile", 0, 100, 95)

            submitted = st.form_submit_button("Apply Params")

            if submitted:
                st.session_state["kmeans"]["params_changed"] = True
                st.session_state["kmeans"]["user_params"] = (k, outlierThreshPercentile)
            else:
                st.session_state["kmeans"]["params_changed"] = False

        if st.session_state["preprocessed_data"] is not None:
            with st.expander("Helper Plots", expanded=True):
                if st.session_state["refresh_helper_plots"]:
                    WCSS_values = utils.generateWCSSValues(st.session_state.preprocessed_data, (1, 10))

                    # Determine a good value for k
                    plot.makeElbowPlot(WCSS_values, "kmeans")
                st.image("kmeans_elbow_plot.png")

def show_dbscan_param_selection():
    if st.session_state.dataset == "Heart Dataset":
        default_eps = 0.7
    else:
        default_eps = 1.4
    with st.expander("DBSCAN Params", expanded=True):
        with st.form("dbscan_params"):
            epsilon = st.number_input("Epsilon", min_value=0.0, value=default_eps)

            if st.session_state["preprocessed_data"] is not None:
                min_samples = st.number_input("Min Samples", min_value=3, value=2 * len(st.session_state.preprocessed_data.columns))
                st.session_state["dbscan"]["helper_plot_param"] = min_samples


            submitted = st.form_submit_button("Apply Params")

            if submitted:
                st.session_state["dbscan"]["params_changed"] = True
                st.session_state["dbscan"]["user_params"] = (epsilon, min_samples)
            else:
                st.session_state["dbscan"]["params_changed"] = False

        # Its only not none if preprocessed data is not none
        if st.session_state["dbscan"]["helper_plot_param"] is not None:
            with st.expander("Helper Plots", expanded=True):
                if st.session_state["refresh_helper_plots"]:
                    kthNearestNeighborDistance = utils.getKthNearestNeighborsDistance(st.session_state.preprocessed_data, st.session_state["dbscan"]["helper_plot_param"])
                    plot.makeElbowPlot(kthNearestNeighborDistance, "dbscan")
                st.image("dbscan_elbow_plot.png")



def show_gmm_param_selection():
    if st.session_state.dataset == "Heart Dataset":
        default_num_components = 8
        default_covariance_type = "diag"
    else:
        default_num_components = 9
        default_covariance_type = "diag"
    with st.expander("GMM Params", expanded=True):

        with st.form("gmm_params"):
            num_components = st.slider("Number of components", 1, 10, default_num_components)
            type_list = ["full", "spherical", "tied", "diag"]
            covariance_type = st.selectbox(
                "Covariance type",
                tuple(type_list),
                index=type_list.index(default_covariance_type))

            outlier_thresh_percentile = st.slider("Outlier threshold percentile", 0, 100, 5)
            submitted = st.form_submit_button("Apply Params")

            if submitted:
                st.session_state["gmm"]["params_changed"] = True
                st.session_state["gmm"]["user_params"] = (num_components, covariance_type, outlier_thresh_percentile)
            else:
                st.session_state["gmm"]["params_changed"] = False

        if st.session_state["preprocessed_data"] is not None:
            with st.expander("Helper Plots", expanded=True):
                if st.session_state["refresh_helper_plots"]:
                    plot.plotGMMModelComparison(st.session_state.preprocessed_data)
                st.image("gmm_model_comparison_plot.png")


def show_alg_param_selection(alg):
    print("param selection alg", alg)
    print("AHHH DATA", st.session_state.preprocessed_data)

    if alg == "kmeans":
        show_kmeans_param_selection()
    elif alg == "dbscan":
        show_dbscan_param_selection()
    elif alg == "gmm":
        show_gmm_param_selection()


def show_cluster_vizualizations(alg):
    print("cluster viz alg", alg)
    st.markdown("#### PCA Projection Colored by Cluster")
    if st.session_state[alg]["user_params"] is not None and st.session_state["preprocessed_data"] is not None:
        print("HIT")
        # pca_df = utils.getPCADf(st.session_state.preprocessed_data)
        if st.session_state[alg]["params_changed"]:
            params = st.session_state[alg]["user_params"]
            clusters = utils.getClusterAssignments(alg, st.session_state.preprocessed_data, *params)

            # This is probably not super efficient but it's probably fine
            st.session_state[alg]["cluster_assgns"] = clusters
            plot.plotPCAWithColors(st.session_state.preprocessed_data, clusters, alg)

        st.image(f"{alg}_pca.png")

    if st.session_state[alg]["cluster_assgns"] is not None and (st.session_state["cluster_summary_all_cols"] or st.session_state["summary_col"]) :
        with st.expander("Cluster Info", expanded=True):
            if st.session_state["cluster_summary_all_cols"]:
                summary_cols = list(st.session_state.preprocessed_data.columns)
            else:
                summary_cols = [st.session_state["summary_col"]]

            for summary_col in summary_cols:
                with st.expander(f"{summary_col}_summary_info", expanded=False):
                    combined_df = st.session_state.raw_data_selected.copy().reset_index(drop=True)
                    combined_df["clusters"] = st.session_state[alg]["cluster_assgns"]

                    st.table(combined_df.groupby("clusters")[summary_col].describe())

                    plotPath = f"{alg}_{summary_col}_violin.png"
                    plot.plotViolinPlot(combined_df, "clusters", summary_col,"clusters", plotPath)
                    st.image(plotPath)



def render_alg_column(alg):
    print("alg column alg", alg)
    proper_name_dict = {
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "gmm": "GMM"
    }
    proper_name = proper_name_dict[alg]
    st.header(proper_name)

    show_alg_param_selection(alg)
    show_cluster_vizualizations(alg)



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


if "initialized" not in st.session_state:
    reset()
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
            st.session_state.dataset_col_dict = data_utils.HEART_COLS
        elif st.session_state.dataset == "Vitals Dataset":
            st.session_state.data_processing_func = data_utils.load_clinical_data
            st.session_state.dataset_col_dict = data_utils.VITAL_COLS

        raw_data_full = st.session_state.data_processing_func(preprocess=False)

        st.session_state.raw_data_full = raw_data_full

    if st.session_state["dataset"] is not None:

        if st.session_state["dataset"] == "Heart Dataset":
            default_col_list = ["sex", "age", "trestbps", "fbs"]
        elif st.session_state["dataset"] == "Vitals Dataset":
            default_col_list = ["sex", "age", "preop_htn", "preop_gluc", "preop_hb", "preop_plt"]

        st.session_state.selected_columns = list(st.multiselect(
            "Columns:",
            list(st.session_state.raw_data_full.columns),
            default=default_col_list
        ))


        if st.toggle("Enable Categorical Class Filtering", value=st.session_state.get("cat_class_filter", False)):
            print("Categorical Class Filtering enabled")

            if st.session_state.dataset == "Heart Dataset":
                categorical_cols = st.session_state.dataset_col_dict["ordinal"] + st.session_state.dataset_col_dict["nominal"] + st.session_state.dataset_col_dict["binary"]

            if st.session_state.dataset == "Vitals Dataset":
                categorical_cols = st.session_state.dataset_col_dict["ordinal"] + st.session_state.dataset_col_dict["nominal"] + st.session_state.dataset_col_dict["binary"]

            categorical_excluding_selected_cols = list(set(categorical_cols) - set(st.session_state.selected_columns))

            filter_col = st.selectbox(
                "Column To Filter With:",
                categorical_excluding_selected_cols,
            )

            categories_to_keep = list(st.multiselect(
                "Categories to Keep:",
                st.session_state.raw_data_full[filter_col].unique(),
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
                st.session_state.categories_to_keep = list(categories_to_keep)
                st.session_state.filter_col = filter_col
                st.session_state.preprocessed_data =  st.session_state.data_processing_func(
                    selected_columns=tuple(st.session_state.selected_columns),
                    filterTuple = (st.session_state.filter_col, st.session_state.categories_to_keep), preprocess = True)
            else:
                st.session_state.preprocessed_data =  st.session_state.data_processing_func(
                    selected_columns=tuple(st.session_state.selected_columns))

            st.session_state.raw_data_selected = st.session_state.raw_data_full.loc[
                st.session_state.preprocessed_data.index].copy()

            if len(st.session_state.preprocessed_data.columns) > 10:
                st.warning("Warning: More than 10 dimensions! May run into issues with distances in high dims.")

            if len(st.session_state.preprocessed_data) < 1000:
                st.warning("Warning: Less than 1000 datapoints!")

if "raw_data_selected" in st.session_state:
    with st.sidebar.expander('Data Information', expanded=True):
        if st.session_state.preprocessed_data is not None:
            df_info_dict = {"nrows": len(st.session_state.preprocessed_data), "ncols": len(st.session_state.preprocessed_data.columns)}
            st.table(df_info_dict)

        # print("RIGHT BEFORE PCA NUM COLS", len(st.session_state.preprocessed_data.columns))
        #
        # with st.expander('Summary Statistics', expanded=True):
        #     st.table(st.session_state.raw_data_selected.describe())

        if st.session_state.preprocessed_data is not None:
            with st.expander('PCA Projection Colored by Variable', expanded=True):
                #combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.raw_data_full["target"]], axis=1)

                datapoint_idxs = st.session_state.preprocessed_data.index
                color_col = st.selectbox("Variable to Color Points", list(st.session_state.raw_data_full.columns), index=0)
                if st.session_state.refresh_helper_plots or st.session_state.last_color != color_col:
                    plot.plotPCAWithColors(st.session_state.preprocessed_data, st.session_state.raw_data_selected[color_col], "sidebar")
                    st.session_state.last_color = color_col


                st.image("sidebar_pca.png")

            with st.expander('(Preprocessed) Dataset Distribution by Variable', expanded=True):
                # combined_df = pd.concat([st.session_state.raw_data_selected, st.session_state.raw_data_full["target"]], axis=1)
                plotName = "distribution_plot.png"

                selected_col = st.selectbox("Column to Plot", list(st.session_state.raw_data_full.columns), index=0)
                if selected_col in st.session_state.dataset_col_dict["ordinal"] or selected_col in st.session_state.dataset_col_dict["binary"]:
                    plot.plotBarPlot(st.session_state.raw_data_selected, selected_col, plotName)
                else:
                    plot.plotHistogram(st.session_state.raw_data_selected, selected_col, plotName)


                st.image(plotName)

            with st.expander('(Preprocessed) Dataset Summary Stats', expanded=True):
                summary_df = st.session_state.raw_data_selected.describe(include="all")
                st.table(summary_df)

            with st.expander('(Preprocessed) Dataset Summary Stats by Variable', expanded=True):
                selected_col = st.selectbox("Column to Plot", list(st.session_state.dataset_col_dict["ordinal"] +
                                                                   st.session_state.dataset_col_dict["nominal"] +
                                                                   st.session_state.dataset_col_dict["binary"]), index=0, key="stats_by_var")
                summary_df = st.session_state.raw_data_selected[st.session_state.dataset_col_dict["ordinal"] +
                                                                st.session_state.dataset_col_dict["nominal"] +
                                                                st.session_state.dataset_col_dict["binary"]].groupby(selected_col).describe(include="all")
                st.table(summary_df)

            #with st.expander('(Preprocessed) Dataset Counts by Categorical Variable', expanded=True):
            # selected_col = st.selectbox("Column to Count", list(st.session_state.raw_data_full.columns), index=0)
            # summary_df = st.session_state.raw_data_selected.describe(include="all")
            # st.table(summary_df)


    if any(st.session_state[alg]["cluster_assgns"] is not None for alg in ["kmeans", "dbscan", "gmm"]):
        with st.sidebar.expander('Cluster Info Var', expanded=True):
            if st.toggle("Generate tables for all columns", value=True):
                st.session_state["cluster_summary_all_cols"] = True
                st.session_state["refresh_summary_plots"] = True
            else:
                st.session_state["cluster_summary_all_cols"] = False
                st.session_state.summary_col = st.selectbox("Column to summarize", list(st.session_state.preprocessed_data.columns), index=0)

        combined_cluster_df = st.session_state.raw_data_full.copy()

        algs_with_clusters = []
        for alg in ["kmeans", "dbscan", "gmm"]:
            if st.session_state[alg]["cluster_assgns"] is not None:
                algs_with_clusters.append(alg)
                combined_cluster_df[alg] = st.session_state[alg]["cluster_assgns"]

        if len(algs_with_clusters) > 0:
            if st.session_state.dataset == "Heart Dataset":
                target = "target"
            elif st.session_state.dataset == "Vitals Dataset":
                target = "emop"

            with st.expander(f"Outlier vs {target}", expanded=True):
                for alg in algs_with_clusters:
                    st.write(f"Outlier vs {alg}")
                    outlier_vs_target = combined_cluster_df[combined_cluster_df[alg] == -1][target].value_counts()
                    st.table(outlier_vs_target)





            with st.expander("Outlier Comparisons", expanded=False):
                if st.session_state["cluster_summary_all_cols"]:
                    for col in list(st.session_state.preprocessed_data.columns):
                        plotName = f"{col}_alg_comparison_plot.png"
                        plot.plotOutlierComparisonViolin(combined_cluster_df, col, plotName)

                        st.image(plotName)
                else:
                    plotName = f"{st.session_state.summary_col}_alg_comparison_plot.png"
                    plot.plotOutlierComparisonViolin(combined_cluster_df, st.session_state.summary_col, plotName)
                    st.image(plotName)

            #with st.expander("Outliers", expanded=False):
            #    st.table(combined_cluster_df)










col1, col2, col3 = st.columns(3)

# KMEANS ---------------------------------------------------------------------
with col1:
    render_alg_column("kmeans")

# DBSCAN ---------------------------------------------------------------------

with col2:
    render_alg_column("dbscan")

# GMM ---------------------------------------------------------------------
with col3:
    render_alg_column("gmm")



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
toggle_button("refresh_summary_plots", False)
print("REFRESH PLOTS AFTER", st.session_state.refresh_helper_plots)
