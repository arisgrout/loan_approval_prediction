#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier


# FUNCTIONS
def fill_null(df):
    """NO NULL dropped all inferred"""

    import copy

    df = copy.deepcopy(df)

    try:
        df["Gender"] = df["Gender"].fillna(value="Unknown")  # new unique class 'Unknown'

        # fill these two with most common value (as significantly skewed) + neither appears very predictive of Loan Approval
        for col in ["Married", "Dependents", "Self_Employed"]:
            df[col] = df[col].fillna(value=(df[col].value_counts().idxmax()))

        # special fillna of LoanAmount (mean of ApplicantIncome bins, since these values reasonably correlate)
        df["ApplicantIncome_logbin"] = pd.cut(np.log(df.ApplicantIncome), bins=5)  # find 5 log bins
        df = df.join(
            df.groupby("ApplicantIncome_logbin").mean().LoanAmount, on="ApplicantIncome_logbin", rsuffix="_binMean"
        )  # join LoanAmount means for each bin to DF
        df["LoanAmount"] = df["LoanAmount"].fillna(
            value=df["LoanAmount_binMean"]
        )  # fillna with binned LoanAmount means

        # VAST majority 512/614, are at indxmax() of 360 days.
        df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df.Loan_Amount_Term.value_counts().idxmax())

        # MASSIVE predictor of Loan Approval, would be a big assumption to assume credit is valid (and give value = 1)
        df["Credit_History"] = df["Credit_History"].fillna(value=0)

        df = df.drop(columns=["ApplicantIncome_logbin", "LoanAmount_binMean"])  # drop helper cols

    except Exception:
        pass

    return df


def encode_cat(df):

    import copy

    df = copy.deepcopy(df)

    try:
        obj = df.columns[df.dtypes == object].tolist()

        # Dummies
        df = df.join(pd.get_dummies(df["Gender"]))
        df = df.join(pd.get_dummies(df["Property_Area"]))

        # Maps
        df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
        df["Dependents"] = df["Dependents"].map({"0": 0, "1": 1, "2": 2, "3+": 3})
        df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
        df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    except Exception:
        pass

    try:
        df = df.drop(columns=["Gender", "Property_Area"])  # rm replaced columns
    except Exception:
        pass

    if isinstance(df, pd.Series):
        df = df.map({"Y": 1, "N": 0})

    return df


def ft_engineer(df):

    import copy

    df = copy.deepcopy(df)

    try:
        df["LoanAmount_log"] = df["LoanAmount"].map(np.log)
        df["TotalIncome"] = df.ApplicantIncome + df.CoapplicantIncome
        df["TotalIncome_log"] = df["TotalIncome"].map(np.log)
        df["ApplicantIncome_log"] = df.ApplicantIncome.map(np.log)  # try log this too

        df = df.drop(columns=["TotalIncome", "LoanAmount", "ApplicantIncome", "Loan_ID"])
    except Exception:
        pass

    return df


df = pd.read_csv("data.csv")
X = df.drop(columns=["Loan_Status"])
y = df.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

replace_null = FunctionTransformer(fill_null)
one_hot = FunctionTransformer(encode_cat)
ft_eng = FunctionTransformer(ft_engineer)

preprocesser = Pipeline(steps=[("null", replace_null), ("cat", one_hot), ("features", ft_eng)])

ft_scaler = Pipeline(
    steps=[
        # ('hot', onehot_cat),
        # ('to_dense', ToDenseTransformer()),
        ("scaler", StandardScaler())
    ]
)

pca_select = PCA(n_components=3)
univ_select = SelectKBest(f_classif, k=5)
ft_selection = FeatureUnion([("pca", pca_select), ("kbest", univ_select)])

base_model = RandomForestClassifier()

rf_noscale = Pipeline(steps=[("preprocess", preprocesser), ("rf", base_model)])

rf_scale = Pipeline(
    steps=[("preprocess", preprocesser), ("scale", ft_scaler), ("ft_select", ft_selection), ("rf", base_model)]
)

params = [
    {
        "rf": [RandomForestClassifier()],
        "rf__bootstrap": [True],
        "rf__max_depth": [30, 40, 50],
        "rf__max_features": ["sqrt"],
        "rf__min_samples_leaf": [1, 2],
        "rf__min_samples_split": [8, 10, 12],
        "rf__n_estimators": list(range(1600, 2000, 20)),
    }
]

# TRY WITHOUT SCALING
rf_ns = GridSearchCV(rf_noscale, params, verbose=3, n_jobs=-1).fit(X_train, y_train)

pickle.dump(rf_ns, open("rf_ns.p", "wb"))

print(rf_ns.best_params_)
print(rf_ns.best_score_)
print(rf_ns.score(X_test, y_test))

with open("rf_ns.txt", "w") as out_file:
    bp = rf_ns.best_params_
    bs = rf_ns.best_score_
    s = rf_ns.score(X_test, y_test)
    out_file.write(str(bp) + "\n" + str(bs) + "\n" + str(s))

# TRY ON SCALED DATA
rf_s = GridSearchCV(rf_scale, params, verbose=3, n_jobs=-1).fit(X_train, y_train)

pickle.dump(rf_s, open("rf_s.p", "wb"))

print(rf_s.best_params_)
print(rf_s.best_score_)
print(rf_s.score(X_test, y_test))

with open("rf_s.txt", "w") as out_file:
    bp = rf_s.best_params_
    bs = rf_s.best_score_
    s = rf_s.score(X_test, y_test)
    out_file.write(str(bp) + "\n" + str(bs) + "\n" + str(s))

