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
# # Imbalanced

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# %%
df = pd.read_excel("data/imbalanced_dataset_wdev.xlsx")

# %%
df

# %% [markdown]
# ## Dummy Data

# %%
# Create an LGBM model
params = {
    'objective': 'regression',
    'linear_tree': True,
}
model = LGBMRegressor(**params)

# Create time serie timestamp indices
ts = np.linspace(0, 10, 100)
X = pd.DataFrame({'ts': ts})

# Generate signal to predict using a simple linear system
y = ts * 6.66

# Train LGBM model
model.fit(X, y)

# Create prediction inputs. Start with timestamp indices
# Shift the initial time range by 0.05 to force interpolation and augment if to force extrapolation
x_preds = pd.DataFrame({'ts': list(ts + 0.05) + [11, 12, 13, 14, 15]})
preds = model.predict(x_preds)
# Plot results.
# LGBM with linear tree can extrapolate
plt.plot(x_preds, x_preds['ts'] * 6.66, label='true values')
plt.plot(x_preds, preds, label='LGBM predicted values')
plt.legend()
# plt.savefig('lgbm_linear.png')
plt.show()

# %% [markdown]
# ## Real Data

# %%
# Create an LGBM model
params = {
    'objective': 'regression',
    'linear_tree': True, #for forecasting
#     'lambda_l1 ': 0.2, 
}
model2 = LGBMRegressor(**params)

# %%
X = df[['Dev']]

# %%
X

# %%
y = df['Loss']

# %%
y.hist()

# %%
df.groupby('Dev').mean()

# %%
# Train LGBM model
model2.fit(X, y)

# %%
y_pred = model2.predict(X)

# %%
y_pred

# %%
res_df = X.copy(deep=True)
res_df['Loss_pred'] = y_pred
res_df.head()

# %%
res_df.groupby('Dev').mean()

# %% [markdown]
# ### Extrapolation

# %%
X_later = pd.DataFrame(np.linspace(1,10,10),columns=['Dev'])

res2_df = X_later.copy(deep=True)

res2_df["Loss_pred"] = model2.predict(X_later)


# %%
res2_df

# %% [markdown]
# **Successful extrapolation**

# %%
plt.plot(res2_df['Dev'],res2_df['Loss_pred'])

# %% [markdown]
# ## Train Test Split w AY

# %%
df

# %%
split_yr =  df["AY"].unique()[-1] # use last as test
print(split_yr)
train_mask = df["AY"] < split_yr

# %%
train_df = df.loc[train_mask,:]
test_df = df.loc[~train_mask,:]

# %%
train_df.groupby("Dev").mean()

# %%
test_df.groupby("Dev").mean()

# %% [markdown]
# ## Split with EvalYr

# %%
df

# %%
split_yr =  df["EvalYr"].unique()[-1] # use last as test
print(split_yr)
train_mask = df["EvalYr"] < split_yr

# %%
train_df = df.loc[train_mask,:]
test_df = df.loc[~train_mask,:]

# %%
train_df.groupby("Dev").mean()

# %%
test_df.groupby("Dev").mean()

# %% [markdown]
# ### remodel w 2 valid yrs

# %%
# test
X = train_df.loc[:,["Dev","AY"]]
y = train_df.loc[:,["Loss"]]

# %%
X_valid = train_df.loc[train_df["EvalYr"] >= (split_yr -2) ,["Dev","AY"]]
y_valid = train_df.loc[train_df["EvalYr"] >= (split_yr -2),["Loss"]]

# %%
# Create an LGBM model
params = {
    'boosting_type': 'gbdt', # reduce overfitting
    'objective': 'tweedie', #imbalanced dataset
    'linear_tree': True, #for forecasting
#     'lambda_l1 ': 0.2, 
    'max_depth': 2, # reduce interactions
    'learning_rate':0.01, # reduce extrapolation variance by keep low?
    'num_iterations':10000, # but learn for a long time
    'early_stopping_round': 30, # as long as there is improvemetn
    'random_state': 42,
    
}
model = LGBMRegressor(**params)

# %%
model.fit(X,y,eval_set= [(X_valid,y_valid)])

# %%
model.predict(test_df[["Dev","AY"]])

# %%
model.predict(test_df[["Dev","AY"]])

# %% [markdown]
# ### Extrapolation v2 dev

# %%
X_later

# %%
X_later = pd.DataFrame({"Dev":np.linspace(1,10,10),"AY": np.ones(10)*2002})

res2_df = X_later.copy(deep=True)

res2_df["Loss_pred"] = model.predict(X_later)


# %%
res2_df

# %% [markdown]
# **Successful extrapolation**

# %%
plt.plot(res2_df['Dev'],res2_df['Loss_pred'])

# %% [markdown]
# ### Extrapolation v3 ay

# %%
X_later = pd.DataFrame([[6,2000],[6,2001],[6,2002],[6,2003],[6,2004],[6,2005],[6,2007]],columns=['Dev',"AY"])

res2_df = X_later.copy(deep=True)

res2_df["Loss_pred"] = model.predict(X_later)


# %%
res2_df

# %% [markdown]
# **Successful extrapolation of dev**

# %%
plt.plot(res2_df['AY'],res2_df['Loss_pred'])

# %% [markdown]
# **Successful extrapolation of AY trend**

# %%
