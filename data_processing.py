import pandas as pd
import sklearn



def load_clinical_data(filePath):
    data = pd.read_csv(filePath)

    print(len(data.columns))
    print("num rows before", len(data))
    #data = data.drop(["preop_ph","preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2"], axis=1)

    # Idk if this will drop too many columns (Yes it will)
    #data = data.dropna()
    data = data.loc[:, ["caseid", "subjectid", "age", "sex", "height", "weight", "bmi", "asa", "emop", "department", "optype", "dx", "opname", "preop_htn", "preop_dm", "preop_ecg", "preop_pft", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]]
    data=data.dropna()


    print("num rows", len(data))
    print(len(data.columns))
    # Age is a string for some reason and has some weird values that can't be turned into floats (yes there are floats in the age for patients younger than 1)
    # So I have to get rid of the weird ones and keep all of them as a float
    data["age"] = pd.to_numeric(data["age"], errors='coerce').dropna()

    # TODO: uncomment these; just starting out really simple for now
    #numeric_cols=["age", "height", "weight", "bmi", "asa", "preop_hb", "preop_plt", "preop_pt", "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb", "preop_ast", "preop_alt", "preop_bun", "preop_cr"]
    #categorical_cols=["department", "optype", "dx", "opname", "preop_ecg", "preop_pft"]
    #binary_cols=["sex"]

    numeric_cols=["age", "height", "weight"]
    categorical_cols=["dx"]
    binary_cols=["sex"]

    # TODO: do I have to reinitialize this every time? idk


    # This isn't working in the pipeline so I have to do it here
    labelBinarizer = sklearn.preprocessing.LabelBinarizer()
    for binary_col in binary_cols:
        data[binary_col] = labelBinarizer.fit_transform(data[binary_col])

    data_preprocessor=get_data_preprocessor(numeric_cols, categorical_cols, binary_cols)

    # TODO: should this be fit or fit_transform
    preprocessed_data = data_preprocessor.fit_transform(data)

    return preprocessed_data


# Basically same code as lab 3 
def get_data_preprocessor(numeric_cols, categorical_cols, binary_cols):
    numeric_pipeline = sklearn.pipeline.Pipeline([
        ("scale", sklearn.preprocessing.StandardScaler())
        ])

    categorical_pipeline = sklearn.pipeline.Pipeline([
        ("onehot", sklearn.preprocessing.OneHotEncoder(
            handle_unknown="error",
            sparse_output=False
            ))     
        ])

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

    return data_preprocessor


if __name__=="__main__":
    clinical_data = load_clinical_data("VitalDB/clinical_data.csv")
    print(clinical_data)




    

    


