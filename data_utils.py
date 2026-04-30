import numpy as np
import pandas as pd
import pandas as df
import sklearn

# Preprocessing functions that need to be a function transformer
# So they can actually be used in the pipelines
#stripString = sklearn.preprocessing.FunctionTransformer(lambda X: np.char.strip(X.astype(str)))
stripString = sklearn.preprocessing.FunctionTransformer(lambda X: pd.DataFrame(X).apply(lambda col: col.str.strip()).values)

lowerString = sklearn.preprocessing.FunctionTransformer(lambda X: pd.DataFrame(X).apply(lambda col: col.str.lower()).values)

#lowerString = sklearn.preprocessing.FunctionTransformer(lambda X: np.char.lower(X.astype(str)))


def load_clinical_data(filePath="data/VitalDB/clinical_data.csv", selected_columns=None, filterTuple=None, preprocess=True, onehot=True):
    data = pd.read_csv(filePath)


    print(len(data.columns))
    print("DEBUGGING PRINTS .........................")
    print("selected_columns:", selected_columns)

    # Columns that are NA often
    #often_na_cols = ["preop_ph","preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2", "lmasize", "iv2", "aline2", "cline1", "cline2", "intraop_ebl", "intraop_uo"]
    #unimportant_cols = ["tubesize", "dltubesize", "aline1"]
    #data = data.drop(columns=often_na_cols + unimportant_cols)


    numeric_cols = ["age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_pt", "preop_aptt",
                    "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun",
                    "preop_cr",
                    "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid", "intraop_ppf",
                    "intraop_ftn",
                    "intraop_rocu", "intraop_vecu", "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca"]

    # numeric_cols=["age", "height", "weight"]
    # categorical_cols=["department", "optype", "dx", "opname", "preop_ecg", "preop_pft"]
    # TODO: figure out what to do with the really big categorical cols
    ordinal_cols = ["cormack", "airway", "preop_ecg"]
    nominal_cols = ["optype", "opname", "dx", "department", "iv1"]
    binary_cols = ["sex", "death_inhosp"]
    data["age"] = pd.to_numeric(data["age"], errors='coerce')

    for col in nominal_cols:
        data[col] = data[col].astype(str).str.strip().str.lower()

    if filterTuple is not None:
        (col_to_filter, categories_to_keep) = filterTuple

        if col_to_filter in ordinal_cols:
            ordinal_cols.remove(col_to_filter)

        if col_to_filter in nominal_cols:
            nominal_cols.remove(col_to_filter)

        data = data[data[col_to_filter].isin(categories_to_keep)]
        # TODO: idk if this gonna work
        data.drop(col_to_filter, axis=1, inplace=True)

    data = data[numeric_cols + nominal_cols + ordinal_cols + binary_cols]
    data = data.dropna()


    assert len(numeric_cols)+len(nominal_cols)+len(ordinal_cols)+len(binary_cols) == len(list(data.columns))


    if preprocess:
        original_index = data.index  # save index as a column
        data = data.reset_index(drop=True)
        print("NEW DATA IDXS AFTER RESET", list(data.index))

        # Do preprocessing steps
        data = preprocess_dataframe(data, numeric_cols, nominal_cols, ordinal_cols, binary_cols, onehot=onehot)


        data.index = original_index

        data = data.dropna()

    # selected_columns is the desired columns
    # This is done after and not before because it won't work with the pipeline if all of the columns aren't passed in
    if selected_columns is not None:
        # Doing starts with instead of the actual col names because when it onehot encodes stuff, it makes each category
        # a diff column starting with the original categorical variable name
        data = data.loc[:, data.columns.str.startswith(selected_columns)]

    print("AHHHHHHHHHHHHHH WTF IS HAPPENING")
    print("COLUMNS", data.columns)

    return data

def load_heart_data(filePath="data/heart.csv", selected_columns=None, filterTuple=None, preprocess=True, onehot=False):
    data = pd.read_csv(filePath)


        # Separate the diff cols based on data type
    # Done like this so they can be passed into the pipeline (do different preprocessing steps to each type of col)
    numeric_cols=["age", "trestbps", "chol", "thalach", "oldpeak"]


    # TODO: double check that these are right
    nominal_cols=[]
    ordinal_cols=["cp", "restecg", "thal", "slope", "ca"]

    binary_cols=["sex", "fbs", "target", "exang"]
    # DATA CLEANING

    for col in nominal_cols:
        data[col] = data[col].astype(str).str.strip().str.lower()

    if filterTuple is not None:
        (col_to_filter, categories_to_keep) = filterTuple

        if col_to_filter in ordinal_cols:
            ordinal_cols.remove(col_to_filter)

        if col_to_filter in nominal_cols:
            nominal_cols.remove(col_to_filter)

        data = data[data[col_to_filter].isin(categories_to_keep)]
        # TODO: idk if this gonna work
        data.drop(col_to_filter, axis=1, inplace=True)

    data = data[numeric_cols + nominal_cols + ordinal_cols + binary_cols]
    data = data.dropna()

    assert len(numeric_cols)+len(nominal_cols)+len(ordinal_cols)+len(binary_cols) == len(list(data.columns))


    if preprocess:

        # Do preprocessing steps
        data = preprocess_dataframe(data, numeric_cols, nominal_cols, ordinal_cols, binary_cols, onehot=onehot)
    else:
        # just clean the data
        for col in nominal_cols:
            data[col] = data[col].astype(str).str.strip().str.lower()

    # selected_columns is the desired columns
    # This is done after and not before because it won't work with the pipeline if all of the columns aren't passed in
    if selected_columns is not None:
        # Doing starts with instead of the actual col names because when it onehot encodes stuff, it makes each category
        # a diff column starting with the original categorical variable name
        data = data.loc[:, data.columns.str.startswith(selected_columns)]

    data = data.dropna()

    return data


# Basically same code as lab 3
# Getting the actual data prerocessor object
def get_data_preprocessor(numeric_cols, nominal_cols, ordinal_cols, binary_cols, onehot):

    # Scale the numeric data
    numeric_pipeline = sklearn.pipeline.Pipeline([
        ("scale", sklearn.preprocessing.StandardScaler())
        ])


    # Onehot encode the nominal data
    nominal_pipeline = sklearn.pipeline.Pipeline([
        ("onehot", sklearn.preprocessing.OneHotEncoder(
            handle_unknown="error",
            sparse_output=False
            ))
        ])


    ordinal_pipeline = sklearn.pipeline.Pipeline([
        ('ordinal_encoding', sklearn.preprocessing.OrdinalEncoder())
    ])



    #binary_pipeline = sklearn.pipeline.Pipeline([
    #    ("label_binarize", sklearn.preprocessing.LabelBinarizer())
    #    ])

    # Put those pipelines into this column transformer thing and this is the data preprocessor
    if onehot is True:
        data_preprocessor = sklearn.compose.ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("nominal", nominal_pipeline, nominal_cols),
                ("ordinal", ordinal_pipeline, ordinal_cols),
                #("binary", binary_pipeline, binary_cols)
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )
    else:
        data_preprocessor = sklearn.compose.ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("ordinal", ordinal_pipeline, ordinal_cols),
                # ("binary", binary_pipeline, binary_cols)
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

    data_preprocessor.set_output(transform="pandas")

    return data_preprocessor

def preprocess_dataframe(dataframe, numeric_cols, nominal_cols, ordinal_cols, binary_cols, onehot=True):
    # This isn't working in the pipeline so I have to do it here
    for binary_col in binary_cols:
        labelBinarizer = sklearn.preprocessing.LabelBinarizer()
        dataframe[binary_col] = labelBinarizer.fit_transform(dataframe[binary_col])

    dataframe_preprocessor = get_data_preprocessor(numeric_cols, nominal_cols, ordinal_cols, binary_cols, onehot=onehot)

    # Get the preprocessed dataframe
    dataframe = dataframe_preprocessor.fit_transform(dataframe)

    return dataframe




if __name__=="__main__":
    clinical_data = load_clinical_data()

    clinical_data.to_csv("cleaned_clinical_data.csv")

    print(clinical_data.dtypes)




    

    


