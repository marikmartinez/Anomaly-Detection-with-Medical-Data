import numpy as np
import pandas as pd
import sklearn



def load_clinical_data(filePath="VitalDB/clinical_data.csv", preprocess=False):
    data = pd.read_csv(filePath)

    print(len(data.columns))
    print("num rows before", len(data))
    #data = data.drop(["preop_ph","preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2"], axis=1)

    # Idk if this will drop too many columns (Yes it will)
    #data = data.dropna()
    data = data.loc[:, ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype", "dx", "opname", "preop_htn", "preop_dm", "preop_ecg", "preop_pft", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]]
    print("UNIQUE DXs", len(np.unique(data["dx"])))
    data = data.dropna()


    print("num rows", len(data))
    print(len(data.columns))
    # Age is a string for some reason and has some weird values that can't be turned into floats (yes there are floats in the age for patients younger than 1)
    # So I have to get rid of the weird ones and keep all of them as a float
    data["age"] = pd.to_numeric(data["age"], errors='coerce').dropna()

    # TODO: uncomment these; just starting out really simple for now
    numeric_cols=["age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]
    categorical_cols=["department", "optype", "dx", "opname", "preop_ecg", "preop_pft"]
    #binary_cols=["sex"]
    #numeric_cols=["age", "height", "weight"]
    #categorical_cols=["dx"]
    binary_cols=["sex"]

    # TODO: do I have to reinitialize this every time? idk


    if preprocess:
        # This isn't working in the pipeline so I have to do it here
        labelBinarizer = sklearn.preprocessing.LabelBinarizer()
        for binary_col in binary_cols:
            data[binary_col] = labelBinarizer.fit_transform(data[binary_col])

        data_preprocessor=get_data_preprocessor(numeric_cols, categorical_cols, binary_cols)


        print("UNIQUE DXs", len(np.unique(data["dx"])))
        # TODO: should this be fit or fit_transform
        data = data_preprocessor.fit_transform(data)
        print("TYPE", type(data))
        print("COLUMNS", list(data.columns))

        # TODO: add more starting strings in tuple if want to add more categorical
    #data_categorical = data.loc[:, data.columns.str.startswith('dx')]

    #data_numeric = data[["age", "height", "weight"]]
    data_numeric = data[["height", "weight", "age"]]

    #data_binary = data[["sex"]]

    # Axis = 1 concatenates along columns, 0 along rows
    #data = pd.concat([data_numeric, data_categorical, data_binary], axis=1)
    #data = pd.concat([data_numeric, data_binary], axis=1)
    data = data_numeric

    return data

# Basically same code as lab 3 
def get_data_preprocessor(numeric_cols, categorical_cols, binary_cols, onehot=True):
    numeric_pipeline = sklearn.pipeline.Pipeline([
        ("scale", sklearn.preprocessing.StandardScaler())
        ])

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


if __name__=="__main__":
    clinical_data = load_clinical_data()

    clinical_data.to_csv("cleaned_clinical_data.csv")

    print(clinical_data.dtypes)




    

    


