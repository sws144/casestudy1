# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: casestudy1
#     language: python
#     name: casestudy1
# ---

# %% [markdown]
# # Case Study

# %% [markdown]
# ## Purpose
#  1. Evalute training dataset to fill in missing data in testing dataset
#  1. Target is **value** column to indicate if a company ($9k) has a claim (100k)
#  1. Expected profit per record = 9k - (value)*100k

# %% [markdown]
# ## Final Conclusion and Proposed Strategy
#  1. Based on the existing data (city and employees), predicted losses are negative (testing_results.csv)
#  - However, if we either more appropriately charge higher risk
#  or stop writing risks > with a level of expected loss (e.g. 0.14), then we can increase profits
#  1. We should also improve dataset by finding status for test set, industry and state by
#  by finding tickers using website like https://bigpicture.io/docs/api/#name-to-domain-api-beta
#  or https://www.klazify.com/category and cross reference with delisted tickers found below
#  1. Details are below

# %% [markdown]
# ## Addendum
#  Streamlit app to understand changes and to iterate on
#  https://sws144-casestudy1-app-streamlit-fz26pv.streamlitapp.com/

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Case-Study" data-toc-modified-id="Case-Study-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Case Study</a></span><ul class="toc-item"><li><span><a href="#Purpose" data-toc-modified-id="Purpose-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Purpose</a></span></li><li><span><a href="#Final-Conclusion-and-Proposed-Strategy" data-toc-modified-id="Final-Conclusion-and-Proposed-Strategy-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Final Conclusion and Proposed Strategy</a></span></li><li><span><a href="#Addendum" data-toc-modified-id="Addendum-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Addendum</a></span></li></ul></li><li><span><a href="#Imports-&amp;-Setup" data-toc-modified-id="Imports-&amp;-Setup-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports &amp; Setup</a></span></li><li><span><a href="#Initial-data-exploration" data-toc-modified-id="Initial-data-exploration-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Initial data exploration</a></span><ul class="toc-item"><li><span><a href="#Conclusion-1" data-toc-modified-id="Conclusion-1-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Conclusion 1</a></span></li></ul></li><li><span><a href="#Naive-model" data-toc-modified-id="Naive-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Naive model</a></span></li><li><span><a href="#Metrics-for-Comparison" data-toc-modified-id="Metrics-for-Comparison-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Metrics for Comparison</a></span></li><li><span><a href="#Support-Functions" data-toc-modified-id="Support-Functions-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Support Functions</a></span></li><li><span><a href="#Try-Gradient-Boosting-with-more-vars-(i.e.-city)" data-toc-modified-id="Try-Gradient-Boosting-with-more-vars-(i.e.-city)-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Try Gradient Boosting with more vars (i.e. city)</a></span><ul class="toc-item"><li><span><a href="#Explain-model" data-toc-modified-id="Explain-model-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Explain model</a></span></li><li><span><a href="#Explainer-plots" data-toc-modified-id="Explainer-plots-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Explainer plots</a></span></li><li><span><a href="#Conclusion-2" data-toc-modified-id="Conclusion-2-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Conclusion 2</a></span></li><li><span><a href="#Pull-Delisted-Stock-Data" data-toc-modified-id="Pull-Delisted-Stock-Data-7.4"><span class="toc-item-num">7.4&nbsp;&nbsp;</span>Pull Delisted Stock Data</a></span></li></ul></li><li><span><a href="#For-Now,-use-GBM-with-Existing-Vars" data-toc-modified-id="For-Now,-use-GBM-with-Existing-Vars-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>For Now, use GBM with Existing Vars</a></span><ul class="toc-item"><li><span><a href="#Explain-smaller-model" data-toc-modified-id="Explain-smaller-model-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Explain smaller model</a></span></li><li><span><a href="#Validation-and-Model-Performanc" data-toc-modified-id="Validation-and-Model-Performanc-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Validation and Model Performanc</a></span></li><li><span><a href="#Explainer-plots" data-toc-modified-id="Explainer-plots-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Explainer plots</a></span></li><li><span><a href="#Conclusion-3" data-toc-modified-id="Conclusion-3-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>Conclusion 3</a></span></li><li><span><a href="#Predict-based-on-model" data-toc-modified-id="Predict-based-on-model-8.5"><span class="toc-item-num">8.5&nbsp;&nbsp;</span>Predict based on model</a></span></li><li><span><a href="#Final-Conclusion-and-Proposed-Strategy" data-toc-modified-id="Final-Conclusion-and-Proposed-Strategy-8.6"><span class="toc-item-num">8.6&nbsp;&nbsp;</span>Final Conclusion and Proposed Strategy</a></span></li></ul></li></ul></div>

# %% [markdown]
# # Imports & Setup

# %%
# imports & settings

import copy
import dill
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mlflow  # for tracking
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

import shap

retune = True
mlflow.sklearn.autolog()
# set experiment
try:
    mlflow.set_experiment("casestudy")
except:
    mlflow.create_experiment("casestudy")

# %%
# Read in data

df = pd.read_csv("data/training.csv")

# %% [markdown]
# # Initial data exploration

# %%
df.head()

# %%
# Read in test data

df_test = pd.read_csv("data/testing.csv")
df_test.head()
df_test.describe(include="all")

# %% [markdown]
#  Test data missing status column

# %%
# Data

df.describe(include="all")

# %% [markdown]
#  Three columns in input, output is binary (0 or 1) and imbalanced with mostly 0
#  There are too many city and website to use natively as categorical. Only employees
#  there is ordering in amount of employees, so consider as ordinal fo later

# %%
# categorical vs numerical vs ordinal to start

categorical_vars = ["status"]  # consider city/website after
numerical_vars = []
ordinal_vars = ["employees"]
target = "value"

# %%
# Investigate initial vars
# as it has reasonable/smaller split

for v in categorical_vars + ordinal_vars:
    print(df.pivot_table("value", aggfunc="mean", index=v))

# %% [markdown]
# ## Conclusion 1
#  1. There may be some signal in status, but this is not available in test set
#  1. Website is one to one, so difficult to use directly
#  1. City is also sparse, but there are frequent data points that maybe we can group
#  1. Employees sign is noisy  no clear trend

# %%
# Use pairplots to quickly see basic relationships for numerical

sns.pairplot(df)

# %%
# Examine records with claims

df_wclaims = df.loc[df["value"] > 0]
df_wclaims.head()

# %%
# Create validation dataset for tuning/checking later

from sklearn.model_selection import train_test_split

# use stratify to get same mean
X_train_train, X_train_valid = train_test_split(
    df, stratify=df["value"], random_state=0
)

print(X_train_train.describe(include="all"))
print(X_train_valid.describe(include="all"))

# %% [markdown]
# # Naive model
#  Using no variables

# %%
# naive model to start

naive_model = sum(df[target]) / df.shape[0]
print(naive_model)

naive_profit = df_test.shape[0] * (9000 - (naive_model) * 100000)
print(f"test set naive profit of {naive_profit}")

# %% [markdown]
# # Metrics for Comparison
#  This is an imbalanced dataset with only two output values, but we can set price as
#  a number, so still treat as a regression problem where we are seeking differentiaation
#  use r2_score as a check to ensure selection is better than global mean

# %%
from sklearn.metrics import r2_score

predictions = np.full((X_train_valid.shape[0],), naive_model)

print(r2_score(X_train_valid[target], predictions))

# which is zero because we've only selected mean
# higher is better (max is 1)

# %% [markdown]
# # Support Functions

# %%
# define gini function for differentiatino
# https://www.kaggle.com/jpopham91/gini-scoring-simple-and-efficient
def gini_normalized(y_true, y_pred, sample_weight=None):
    # check and get number of samples
    assert (
        np.array(y_true).shape == np.array(y_pred).shape
    ), "y_true and y_pred need to have same shape"
    n_samples = np.array(y_true).shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    if sample_weight == None:
        sample_weight = np.ones(n_samples)

    arr = np.array([y_true, y_pred, sample_weight]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]  # true col sorted by true
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]  # true col sorted by pred

    true_order_wgts = arr[arr[:, 0].argsort()][::-1, 2]
    pred_order_wgts = arr[arr[:, 0].argsort()][::-1, 2]

    # get Lorenz curves
    L_true = np.cumsum(np.multiply(true_order, true_order_wgts)) / np.sum(
        np.dot(true_order, true_order_wgts)
    )
    L_pred = np.cumsum(np.multiply(pred_order, pred_order_wgts)) / np.sum(
        np.multiply(pred_order, pred_order_wgts)
    )
    L_ones = np.multiply(np.linspace(1 / n_samples, 1, n_samples), pred_order_wgts)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true


assert gini_normalized([1, 2], [1, 2]) == 1, "perfect matching should have gini of 1"

print(f"starting gini {gini_normalized(X_train_valid[target], predictions)}")

# higher is better (1 is max)

# %%
# Other functions


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    formula,
    weights=None,
    tweedie_powers=None,
):
    """
    Evaluate an estimator on train and test sets with different metrics
    Requires active run on mlflow and estimator with .predict method
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, auc, r2_score
    from functools import partial
    from sklearn.metrics import mean_tweedie_deviance

    mlflow.set_tag("run_id", mlflow.active_run().info.run_id)
    mlflow.log_params({"formula": formula})

    metrics = [
        # ("default score", None),   # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
        ("gini", gini_normalized),
        ("r2", r2_score),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    res = {}
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        if weights != None:
            y, _weights = df[target], df[weights]
        else:
            y, _weights = df[target], None

        if isinstance(estimator, tuple) and len(estimator) == 2:
            # Score the model consisting of the product of frequency and
            # severity models.
            est_freq, est_sev = estimator
            y_pred = est_freq.predict(X) * est_sev.predict(X)
        elif "h2o" in str(type(estimator)):
            y_pred = (
                estimator.predict(h2o.H2OFrame(X)).as_data_frame().to_numpy().ravel()
            )  # ensure 1D array
        else:
            y_pred = estimator.predict(X)

        for score_label, metric in metrics:

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res[score_label + "_" + subset_label] = score

    return res


def to_categorical(x):
    return x.astype("category")


def to_float(x):
    return x.astype(float)


# %% [markdown]
# # Try Gradient Boosting with more vars (i.e. city)

# %%
# find top cities

city_counts = df["city"].value_counts()

top_city = ["NA"] + list(
    city_counts[city_counts > 30].index
)  # 30 is somewhat arbitrary, tied to idea that ~30 needed for credibilty of a statistic

# %%
# build model

numeric_features = []
categorical_features = [
    "status",
]  # df is missing "status","city", "website" are too sparse
employee_feature = ["employees"]
city_feature = ["city"]
all_features = numeric_features + categorical_features + employee_feature + city_feature
# order matters

mlflow.end_run()
mlflow.start_run(run_name="sklearn_hgbm")

numeric_transformer = Pipeline(
    steps=[
        ("to_float", FunctionTransformer(func=to_float)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("to_categorical", FunctionTransformer(func=to_categorical)),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        ),
    ]
)

employee_transformer = Pipeline(
    steps=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[
                    [
                        "1-10",
                        "11-50",
                        "51-100",
                        "101-250",
                        "251-500",
                        "501-1000",
                        "1001-5000",
                        "5001-10000",
                        "10001+",
                    ]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
        ),
    ]
)

city_transformer = Pipeline(
    steps=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[top_city],
                handle_unknown="use_encoded_value",
                unknown_value=len(top_city),
            ),
        ),
    ]
)


# based on variable order
categorical_mask = (
    [False] * len(numeric_features)
    + [True] * len(categorical_features)
    + [True] * len(employee_feature)
    + [True] * len(city_feature)
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("emp", employee_transformer, employee_feature),
        ("city", city_transformer, city_feature),
    ],
    remainder="drop",
)


# tuning
if retune:
    gini_scorer = make_scorer(gini_normalized, greater_is_better=True)

    # use hyperopt package with to better search
    # https://github.com/hyperopt/hyperopt/wiki/FMin
    # use userdefined Gini, as it measures differentiation more
    def objective_gbr(params):
        "objective_gbr function for hyper opt, params is dict of params for mdl"
        mlflow.start_run(nested=True)
        parameters = {}
        for k in params:
            parameters[k] = params[k]
        mdl = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "estimator",
                    HistGradientBoostingRegressor(
                        random_state=0,
                        **parameters,
                        categorical_features=categorical_mask,
                    ),
                ),
            ]
        )
        score = cross_val_score(
            mdl, df, df[target].squeeze(), scoring=gini_scorer, cv=5
        ).mean()
        print("Gini {:.3f} params {}".format(score, parameters))
        mlflow.end_run()
        return score

    # need to match estimator
    space = {
        # low # high # number of choices
        "learning_rate": hp.uniform("learning_rate", 0.1, 1),
        "max_depth": hp.quniform("max_depth", 2, 4, 2),
        "loss": "poisson",  # implies log-link, which is desired with non-negative data
    }

    best_params = fmin(fn=objective_gbr, space=space, algo=tpe.suggest, max_evals=5)

    for key in best_params.keys():
        if int(best_params[key]) == best_params[key]:
            best_params[key] = int(best_params[key])

    print("Hyperopt estimated optimum {}".format(best_params))

else:
    # ran earlier
    best_params = {
        "learning_rate": 0.6118176373029943,
        "max_depth": 2,
        "loss": "poisson",
    }

mdl = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "estimator",
            HistGradientBoostingRegressor(
                random_state=0, **best_params, categorical_features=categorical_mask
            ),
        ),
    ]
)
# reg = GradientBoostingRegressor(random_state=0)
mdl.fit(df[all_features], df[target].squeeze())

# log with validation
# log_w_validate(y_test, y_pred, formula)
res = score_estimator(
    mdl, X_train_train, X_train_valid, X_train_train, X_train_valid, target, ""
)

mlflow.log_metrics(res)
mlflow.set_tag("target", target)

# addition artifacts
# visualize a single tree
# Get a tree
# sub_tree_1 = reg.estimators_[0, 0]  # pull first 1 estimator, actual regressor vs array

# tree.plot_tree(sub_tree_1, feature_names=list(X_train.columns), filled=True, fontsize=7)

# plt.tight_layout()
# plt.savefig("tree_plot1.png", bbox_inches="tight")
# plt.show()

# mlflow.log_artifact("tree_plot1.png")

# save requirements
os.system("pipenv lock --keep-outdated -d -r > requirements.txt")
mlflow.log_artifact("requirements.txt")

# save categorical dict values
cat_dict = {}
for c in categorical_features + city_feature + employee_feature:
    cat_dict[c] = list(df[c].unique())

with open(f"cat_dict.pkl", "wb") as handle:
    dill.dump(cat_dict, handle, recurse=True)

mlflow.log_artifact(f"cat_dict.pkl")

os.remove(f"cat_dict.pkl")

mlflow.end_run()

# %% [markdown]
# ## Explain model

# %%
explainer = shap.Explainer(mdl[-1])

# fix save expected value
if len(explainer.expected_value.shape) > 0:
    ev = explainer.expected_value[0]
    explainer.expected_value = ev

shap_obj = explainer(mdl[0].transform(df))


var = "status"

# shap_obj, requires column transformer in step position 0 ,
# categorical in position 1

# def update_shap_obj(shap_obj, X_train, encoder):

rename_list = [
    (
        1,  # pipeline order
        1,  # order of ordinalencoder
        categorical_features,  # column name
    ),
    (
        2,  # pipeline order
        0,  # order of ordinalencoder
        employee_feature,  # column name
    ),
    (
        3,  # pipeline order
        0,  # order of ordinalencoder
        city_feature,  # column name
    ),
]


shap_obj.feature_names = list(all_features)
shap_cat = copy.deepcopy(shap_obj)
shap_cat.data = np.array(shap_obj.data, dtype="object")
for name_tuple in rename_list:

    trans_idx = name_tuple[0]
    # ordinal_idx = name_tuple[1]
    cat = name_tuple[2]

    # categorical_names = list(df.select_dtypes(include=["object"]).columns)
    col_idx = list(np.where(np.isin(shap_obj.feature_names, cat))[0])

    # fix categorical names
    res_arr = (
        mdl[0]
        .transformers_[trans_idx][1][-1]
        .inverse_transform(pd.DataFrame(shap_cat.data[:, col_idx], columns=[cat]))
    )

    # update shap col
    for i, loc in enumerate(col_idx):
        shap_cat.data[:, loc] = res_arr[:, i]

# new_dtype = "object"
# res_arr.astype(
#     [(col, new_dtype) if d[0] in categorical_names else d for d in res_arr.dtype.descr]
# )

# col_idx = shap_obj.feature_names.index(var)
# ord_encode_idx = mdl[0].transformers_[1][2].index(var)

# %% [markdown]
# ## Explainer plots
#  overall plot

# %%
shap.plots.beeswarm(shap_cat)

# individual plots
for var in all_features:

    fig, ax = plt.subplots()

    shap.plots.scatter(shap_obj[:, var], ax=ax, show=False, color=shap_obj)

    # find which transformer has this feature
    trans_idx = 0
    for i, trans in enumerate(mdl[0].transformers_):
        if var in trans[-1]:
            trans_idx = i
            var_idx = trans[-1].index(var)

    # replace labels with orig
    if var not in numeric_features:
        # get integer labels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        orig_list = ax.get_xticks()
        new_list = np.insert(
            mdl[0].transformers_[trans_idx][1][-1].categories_[var_idx],
            0,
            "Unknown",
        )

        for i in range(len(orig_list) - len(new_list)):
            new_list = np.append(new_list, orig_list[i + len(new_list)])

        ax.set_xticks(orig_list)
        if len(orig_list) > 5:
            ax.set_xticklabels(new_list, rotation=90)
        else:
            ax.set_xticklabels(new_list)

    plt.show()

shap.plots.waterfall(shap_cat[0])

# %% [markdown]
# ## Conclusion 2
#  For the training dataset, and assuming test dataset is similar
#  1. status, especially status=delisted records, have greater chance of claim
#  1. smallest employees have greater chance as well
#  1. city is least predictive, overall
#  1. test data is missing status, but can try to use website to see if delisted

# %% [markdown]
# ## Pull Delisted Stock Data
#  https://www.alphavantage.co/documentation/  for free API key
#  pulled listing_status.csv from as of 2021
#  https://www.alphavantage.co/query?function=LISTING_STATUS&date=2022-06-10&state=delisted&apikey=REPLACEME

# %% [markdown]
# # For Now, use GBM with Existing Vars

# %%
numeric_features = []
categorical_features = [
    # "status",
]  # df is missing "status","city", "website" are too sparse
employee_feature = ["employees"]
city_feature = ["city"]
all_features = numeric_features + categorical_features + employee_feature + city_feature
# order matters

mlflow.end_run()
mlflow.start_run(run_name="sklearn_hgbm")

numeric_transformer = Pipeline(
    steps=[
        ("to_float", FunctionTransformer(func=to_float)),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("to_categorical", FunctionTransformer(func=to_categorical)),
        (
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
        ),
    ]
)

employee_transformer = Pipeline(
    steps=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[
                    [
                        "1-10",
                        "11-50",
                        "51-100",
                        "101-250",
                        "251-500",
                        "501-1000",
                        "1001-5000",
                        "5001-10000",
                        "10001+",
                    ]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
        ),
    ]
)

city_transformer = Pipeline(
    steps=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[top_city],
                handle_unknown="use_encoded_value",
                unknown_value=len(top_city),
            ),
        ),
    ]
)


# based on variable order
categorical_mask = (
    [False] * len(numeric_features)
    + [True] * len(categorical_features)
    + [True] * len(employee_feature)
    + [True] * len(city_feature)
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        # ("cat", categorical_transformer, categorical_features),
        ("emp", employee_transformer, employee_feature),
        ("city", city_transformer, city_feature),
    ],
    remainder="drop",
)


# tuning
if retune:
    gini_scorer = make_scorer(gini_normalized, greater_is_better=True)

    # use hyperopt package with to better search
    # https://github.com/hyperopt/hyperopt/wiki/FMin
    # use userdefined Gini, as it measures differentiation more
    def objective_gbr(params):
        "objective_gbr function for hyper opt, params is dict of params for mdl"
        mlflow.start_run(nested=True)
        parameters = {}
        for k in params:
            parameters[k] = params[k]
        mdl = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "estimator",
                    HistGradientBoostingRegressor(
                        random_state=0,
                        **parameters,
                        categorical_features=categorical_mask,
                    ),
                ),
            ]
        )
        score = cross_val_score(
            mdl, df, df[target].squeeze(), scoring=gini_scorer, cv=5
        ).mean()
        print("Gini {:.3f} params {}".format(score, parameters))
        mlflow.end_run()
        return score

    # need to match estimator
    space = {
        # low # high # number of choices
        "learning_rate": hp.uniform("learning_rate", 0.1, 1),
        "max_depth": hp.quniform("max_depth", 2, 4, 2),
        "loss": "squared_error",  # poisson did not have good results
    }

    best_params = fmin(fn=objective_gbr, space=space, algo=tpe.suggest, max_evals=5)

    for key in best_params.keys():
        if int(best_params[key]) == best_params[key]:
            best_params[key] = int(best_params[key])

    print("Hyperopt estimated optimum {}".format(best_params))

else:
    best_params = {
        "learning_rate": 0.26984205978817966,
        "max_depth": 4,
        "loss": "squared_error",  # poisson did not have good results
    }

mdl = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "estimator",
            HistGradientBoostingRegressor(
                random_state=0, **best_params, categorical_features=categorical_mask
            ),
        ),
    ]
)
# reg = GradientBoostingRegressor(random_state=0)
mdl.fit(df[all_features], df[target].squeeze())

# log with validation
# log_w_validate(y_test, y_pred, formula)
res = score_estimator(
    mdl, X_train_train, X_train_valid, X_train_train, X_train_valid, target, ""
)

mlflow.log_metrics(res)
mlflow.set_tag("target", target)

# addition artifacts
# visualize a single tree
# Get a tree
# sub_tree_1 = reg.estimators_[0, 0]  # pull first 1 estimator, actual regressor vs array

# tree.plot_tree(sub_tree_1, feature_names=list(X_train.columns), filled=True, fontsize=7)

# plt.tight_layout()
# plt.savefig("tree_plot1.png", bbox_inches="tight")
# plt.show()

# mlflow.log_artifact("tree_plot1.png")

# save requirements
os.system("pipenv lock --keep-outdated -d -r > requirements.txt")
mlflow.log_artifact("requirements.txt")

# save categorical dict values
cat_dict = {}
for c in categorical_features + city_feature + employee_feature:
    cat_dict[c] = list(df[c].unique())

with open(f"cat_dict.pkl", "wb") as handle:
    dill.dump(cat_dict, handle, recurse=True)

mlflow.log_artifact(f"cat_dict.pkl")

os.remove(f"cat_dict.pkl")

# mlflow.end_run()

# %% [markdown]
# ## Explain smaller model

# %%
explainer = shap.Explainer(mdl[-1])

# fix save expected value
if len(explainer.expected_value.shape) > 0:
    ev = explainer.expected_value[0]
    explainer.expected_value = ev

shap_obj = explainer(mdl[0].transform(df))


var = "status"

# shap_obj, requires column transformer in step position 0 ,
# categorical in position 1

# def update_shap_obj(shap_obj, X_train, encoder):

rename_list = [
    # (
    #     1,  # pipeline order
    #     1,  # order of ordinalencoder
    #     categorical_features,  # column name
    # ),
    (
        1,  # pipeline order
        0,  # order of ordinalencoder
        employee_feature,  # column name
    ),
    (
        2,  # pipeline order
        0,  # order of ordinalencoder
        city_feature,  # column name
    ),
]


shap_obj.feature_names = list(all_features)
shap_cat = copy.deepcopy(shap_obj)
shap_cat.data = np.array(shap_obj.data, dtype="object")
for name_tuple in rename_list:

    trans_idx = name_tuple[0]
    # ordinal_idx = name_tuple[1]
    cat = name_tuple[2]

    # categorical_names = list(df.select_dtypes(include=["object"]).columns)
    col_idx = list(np.where(np.isin(shap_obj.feature_names, cat))[0])

    # fix categorical names
    res_arr = (
        mdl[0]
        .transformers_[trans_idx][1][-1]
        .inverse_transform(pd.DataFrame(shap_cat.data[:, col_idx], columns=[cat]))
    )

    # update shap col
    for i, loc in enumerate(col_idx):
        shap_cat.data[:, loc] = res_arr[:, i]

# new_dtype = "object"
# res_arr.astype(
#     [(col, new_dtype) if d[0] in categorical_names else d for d in res_arr.dtype.descr]
# )

# col_idx = shap_obj.feature_names.index(var)
# ord_encode_idx = mdl[0].transformers_[1][2].index(var)

# %% [markdown]
# ## Validation and Model Performanc

# %%
res

# %% [markdown]
# Both gini and r2 are positive, so model is better than naive model

# %%
true_value =  df[target].squeeze()
predicted_value = mdl.predict(df)

plt.figure(figsize=(5,5))
plt.scatter(predicted_value, true_value, c='crimson')

p1 = max(max(predicted_value), max(true_value))
p2 = min(min(predicted_value), min(true_value))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.ylabel('True Values', fontsize=15)
plt.xlabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

# %%
# quantile plot
# https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test

fig, ax = plt.subplots()

int_labels, col_bins = pd.cut(predicted_value, bins=10, retbins=True, labels=False, precision=4)

shap_col_df = pd.DataFrame(
    {
        "var": col_bins[int_labels],
        "actual": true_value,
        "predicted": predicted_value,
    }
)

shap_col_grp_df = shap_col_df.groupby("var").mean()

shap_col_grp_df.plot(ax=ax,marker='o')
ax.set_xlabel('predicted_bins')


# %% [markdown]
# ## Explainer plots
#  overall plot

# %%
shap.plots.beeswarm(shap_cat)

# individual plots
for var in all_features:

    fig, ax = plt.subplots()

    shap.plots.scatter(shap_obj[:, var], ax=ax, show=False, color=shap_obj)

    # find which transformer has this feature
    trans_idx = 0
    for i, trans in enumerate(mdl[0].transformers_):
        if var in trans[-1]:
            trans_idx = i
            var_idx = trans[-1].index(var)

    # replace labels with orig
    if var not in numeric_features:
        # get integer labels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        orig_list = ax.get_xticks()
        new_list = np.insert(
            mdl[0].transformers_[trans_idx][1][-1].categories_[var_idx],
            0,
            "Unknown",
        )

        for i in range(len(orig_list) - len(new_list)):
            new_list = np.append(new_list, orig_list[i + len(new_list)])

        # average change
        shap_col_df = pd.DataFrame(
            {
                var: shap_obj[:, var].data,
                "expected_value_shap": (
                    shap_obj[:, var].values  # + shap_cat[:, var].base_values
                ),
            }
        )
        shap_col_grp_df = shap_col_df.groupby(var).mean()
        shap_col_grp_df.plot(ax=ax)

        # rotate as necessary
        ax.set_xticks(orig_list)
        if len(orig_list) > 5:
            ax.set_xticklabels(new_list, rotation=90)
        else:
            ax.set_xticklabels(new_list)

    plt.show()

# sample
shap.plots.waterfall(shap_cat[0])

with open(f"explainer.pkl", "wb") as handle:
    dill.dump(explainer, handle, recurse=True)

mlflow.log_artifact(f"explainer.pkl")

os.remove(f"explainer.pkl")

mlflow.end_run()

# %% [markdown]
# ## Conclusion 3
#  1. based on the simpler model, smallest employees and Chicago/Palo Alto have slightly higher risk vs New York
#  1. Some of this may be due to correlation with delisted, so is something to consider

# %% [markdown]
# ## Predict based on model

# %%
df_test["predicted_value"] = mdl.predict(df_test)

test_profit_perco = 9000 - np.average(df_test["predicted_value"]) * 100000

print(f"estimated profit per company is {test_profit_perco}")

df_test.to_csv(r"data/testing_results.csv")

# %% [markdown]
# ## Final Conclusion and Proposed Strategy
#  See top

# %%
#
include_idx = (
    df_test["predicted_value"] < 0.13
)  # selected based on business/growth goals
num_pols = sum(include_idx)
test_profit_filtered = (
    9000 * num_pols - sum(df_test.loc[include_idx, "predicted_value"]) * 100000
)
test_profit_filtered_avg = test_profit_filtered / num_pols

print(f"out of original {df_test.shape[0]} records, included {num_pols}")
print(f"total profit with new strat {test_profit_filtered}")

print(f"avg profit {test_profit_filtered_avg}")

# %% [markdown]
#  The final level of exclusion or rate change should consider - expected new business,
#  how confident we are in individual rate change factors, and expected business that is lost

# %%
#
