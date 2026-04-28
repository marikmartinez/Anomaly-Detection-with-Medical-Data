import numpy as np
import pandas as pd
import pandas as df
import sklearn

def load_clinical_data(filePath="data/VitalDB/clinical_data.csv", selected_columns=None, preprocess=True):
    data = pd.read_csv(filePath)

    print(len(data.columns))
    print("DEBUGGING num rows before", len(data))

    # Idk if this will drop too many columns (Yes it will)
    #data = data.dropna()
    # TODO: wtf is going on here D:


    # Columns that are NA often
    often_na_cols = ["preop_ph","preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2", "lmasize", "iv2", "aline2", "cline1", "cline2", "intraop_ebl", "intraop_uo"]
    unimportant_cols = ["tubesize", "dltubesize", "aline1"]
    data = data.drop(columns=often_na_cols + unimportant_cols)
    # TODO: remove this later after debugging
    #data = data[["age", "height", "weight", "sex"]]
    print("num rows", len(data))
    data = data.dropna()
    print("DROPPED NAS")
    print("num rows", len(data))
    print("NUM COLS", len(data.columns))

    # TODO: uncomment these; just starting out really simple for now
    numeric_cols=["age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_pt", "preop_aptt",
                  "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr",
                  "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid", "intraop_ppf", "intraop_ftn",
                  "intraop_rocu", "intraop_vecu", "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca"]

    #numeric_cols=["age", "height", "weight"]
    #categorical_cols=["department", "optype", "dx", "opname", "preop_ecg", "preop_pft"]
    # TODO: figure out what to do with the really big categorical cols
    categorical_cols=["cormack", "airway", "iv1", "preop_ecg"]
    #binary_cols=["sex"]
    #numeric_cols=["age", "height", "weight"]
    #categorical_cols=["dx"]
    binary_cols=["sex"]

    # Age is a string for some reason and has some weird values that can't be turned into floats (yes there are floats in the age for patients younger than 1)
    # So I have to get rid of the weird ones and keep all of them as a float
    data["age"] = pd.to_numeric(data["age"], errors='coerce')

    #data = data.loc[:, ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype", "dx", "opname", "preop_htn", "preop_dm", "preop_ecg", "preop_pft", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]]
    #print("UNIQUE DXs", len(np.unique(data["dx"])))

    # TODO: do I have to reinitialize this every time? idk

    if preprocess:
        data = preprocess_dataframe(data, numeric_cols, categorical_cols, binary_cols, onehot=True)

    if selected_columns is not None:
        data = data.loc[:, data.columns.str.startswith(selected_columns)]

    # DROPPING NAS AGAIN

    print("AHHH NUM COLS", len(data.columns))
    print("AHH NUM ROWS", len(data))
    print("DROPPING NAS AGAIN")
    data = data.dropna()
    print("AHHH NUM COLS", len(data.columns))
    print("AHH NUM ROWS", len(data))

    return data

def load_heart_data(filePath="data/heart.csv", selected_columns=None, preprocess=True):
    data = pd.read_csv(filePath)
    data = data.dropna()

    # Separate the diff cols based on data type
    # Done like this so they can be passed into the pipeline (do different preprocessing steps to each type of col)
    numeric_cols=["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_cols=["cp", "restecg", "thal", "slope", "ca"]
    binary_cols=["sex", "fbs", "target", "exang"]

    assert len(numeric_cols)+len(categorical_cols)+len(binary_cols) == len(list(data.columns))

    if preprocess:

        # Do preprocessing steps
        data = preprocess_dataframe(data, numeric_cols, categorical_cols, binary_cols, onehot=True)

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
def get_data_preprocessor(numeric_cols, categorical_cols, binary_cols, onehot=True):

    # Scale the numeric data
    numeric_pipeline = sklearn.pipeline.Pipeline([
        ("scale", sklearn.preprocessing.StandardScaler())
        ])


    # Onehot encode the categorical data
    if onehot is True:
        categorical_pipeline = sklearn.pipeline.Pipeline([
            ("onehot", sklearn.preprocessing.OneHotEncoder(
                handle_unknown="error",
                sparse_output=False
                ))
            ])
    else:
        categorical_pipeline='passthrough'

    #binary_pipeline = sklearn.pipeline.Pipeline([
    #    ("label_binarize", sklearn.preprocessing.LabelBinarizer())
    #    ])

    # Put those pipelines into this column transformer thing and this is the data preprocessor
    data_preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
            #("binary", binary_pipeline, binary_cols)
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    data_preprocessor.set_output(transform="pandas")

    return data_preprocessor

def preprocess_dataframe(dataframe, numeric_cols, categorical_cols, binary_cols, onehot=True):
    # This isn't working in the pipeline so I have to do it here
    labelBinarizer = sklearn.preprocessing.LabelBinarizer()
    for binary_col in binary_cols:
        dataframe[binary_col] = labelBinarizer.fit_transform(dataframe[binary_col])

    dataframe_preprocessor = get_data_preprocessor(numeric_cols, categorical_cols, binary_cols)

    # Get the preprocessed dataframe
    dataframe = dataframe_preprocessor.fit_transform(dataframe)

    return dataframe




if __name__=="__main__":
    clinical_data = load_clinical_data()

    clinical_data.to_csv("cleaned_clinical_data.csv")

    print(clinical_data.dtypes)




    

    


