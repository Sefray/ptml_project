#!/usr/bin/env python
# coding: utf-8

# ---
# PTML Project: NASA Nearest Object from Earth
# ---
# 
# # Introduction
# 
# Authors:
# 
# - Baptiste Bourdet
# 
# - Philippe Bernet
# 
# - Marius Dubosc
# 
# - Hugo Levy
# 
# The dataset comes from Kaggle: https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects.
# 
# In this report we will try to analyze this data and compute models of both supervised and unsupervised learning to respond to a problem that was highlighted in many movies of science fiction: are any objects currently in orbit a danger to either satellites or earth.

# In[ ]:


import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ## Loading the data
# 
# First we need to load the data and analyse the different component it is made out of.

# In[ ]:


df = pd.read_csv("neo.csv")
# set the id column to be a category
df["id"] = df["id"].astype(object)
df["orbiting_body"] = df["orbiting_body"].astype("category")
df


# The data is composed of the following columns:
# 
# - id: index number
# 
# - name: name of the object
# 
# - est_dimater_min: smallest size of the object in km
# 
# - est_dimater_max: biggest size of the object in km
# 
# - relative_velocity: velocity relative to Earth in km/h
# 
# - orbiting_body: the body the object is orbiting (Earth, Sun, the Moon ...)
# 
# - sentry_object: whether or not the object is tracked by the sentry system of the nasa
# 
# - absolute_magnitude: visibility index, the smaller it is, the brigther the object it, the magnitude of the sun is -27 for example
# 
# - **hazardous**: whether or not the object is considerer a potential threat by the nasa, it is this column we will want to monitor

# Let's start by checking Null values

# In[ ]:


df.info()


# # Analysis
# 
# ## Basic statistics
# 
# The start of the analysis will be purely on the statistic to gain a batter comprehension of the data.

# In[ ]:


print(f"Length of the dataset is {len(df)}")


# In[ ]:


df.hazardous.value_counts().plot(kind='bar')
plt.title('Hazardous')
plt.show()


# In[ ]:


print("Summary of all numerical values :")
df.describe()


# In[ ]:


df.hist(bins=10, figsize=(10, 7))
pass


# In[ ]:


print("Visualization of categorical values :")
print(df.describe(include=["category", "bool"]))


# As we can see we have only one `orbiting body` and only one value for the `sentry`. Those two columns can therefore be removed from the dataset safely.
# The final column, the one we want to predict with the models, seems to have 10% of dangerous objects, the outliers we will try to identify.

# In[ ]:


df_stat = df.drop(columns=["sentry_object", "orbiting_body"])
pass


# In[ ]:


fig= sns.pairplot(df_stat[["est_diameter_min","est_diameter_max","relative_velocity","miss_distance","absolute_magnitude"]])


# In[ ]:


plt.scatter(df_stat.est_diameter_min, df.est_diameter_max)


# We see that est_diameter_min and est_diamater_max are completly linearly correlated. We can therefore remove one.

# In[ ]:


df_stat = df.drop(['est_diameter_min'], axis=1)


# In[ ]:


df_stat


# ## Correlation
# 
# The correlation matrix of our data is the following:

# In[ ]:


def dipslay_corr(df, corr):
    f = plt.figure(figsize=(8, 6))
    plt.matshow(corr, fignum=f.number)
    plt.xticks(
        range(df_stat.select_dtypes(["number"]).shape[1]),
        df_stat.select_dtypes(["number"]).columns,
        fontsize=9,
        rotation=45,
    )
    plt.yticks(
        range(df_stat.select_dtypes(["number"]).shape[1]),
        df_stat.select_dtypes(["number"]).columns,
        fontsize=9,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=9)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


# Matrix de correlation
corr = df_stat.corr()
dipslay_corr(df_stat, corr)


# In[ ]:


df_stat


# We can see in the matrix that the magnitude is negatively correlated with mos of the other variables. This implies that objects with a high magntiude, meaning not very visible objects, are usually smaller and slower. They also more importently do not consistute a threat seeing how the magnitude is negatively correlated with the hazardous.
# 
# When computing the correlation matrix only for the hazardous objects, we can see an even greater correlation between the magnitude and the size.

# In[ ]:


# Matrix de correlation
df_hazard_only = df_stat[df_stat["hazardous"] == True].drop(columns=["hazardous"])
corr_only_hazardous = df_hazard_only.corr()
dipslay_corr(df_hazard_only, corr_only_hazardous)


# ### Dimension Reduction
# 
# It seems some variables are more important than other. Let's see how much information we can keep while reducing the dimension.

# In[ ]:


X = df.drop(columns=["hazardous"])
Y = df["hazardous"]


# In[ ]:


X


# In[ ]:


X = df[
    [
        "est_diameter_max",
        "relative_velocity",
        "miss_distance",
        "absolute_magnitude",
    ]
]
y = df["hazardous"]


# Scale the X
scaler = StandardScaler()
scaler.fit(X)
scaled = scaler.transform(X)

# Obtain principal components
pca = PCA().fit(scaled)

pc = pca.transform(scaled)
pc1 = pc[:, 0]
pc2 = pc[:, 1]

# Plot principal components
plt.figure(figsize=(10, 10))

colour = ["#ff2121" if e else "#2176ff" for e in y]
plt.scatter(pc1, pc2, c=colour, edgecolors="#000000")
plt.show()
pass


# In[ ]:


labels = ["PC1", "PC2", "PC3", "PC4"]

plt.figure(figsize=(15, 7))
plt.bar(labels, pca.explained_variance_)
plt.xlabel("Pricipal Component")
plt.ylabel("Proportion of Variance Explained")
plt.show()
pass


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


for i in range(len(pca.explained_variance_ratio_)):
	ratio = pca.explained_variance_ratio_[:i+1].sum() /pca.explained_variance_ratio_.sum() * 100
	print(f"{ratio}% variance explained for {i} axes")


# We see that two axes are enough to explain 76% of the variance. This could be a way to better represent and explain the data for the models.

# ## 3D representation
# 
# Another approach to visualization is to see how the different columns interact with each other.

# In[ ]:


def show3D_data(df, x, y, z):
    """Show the data in 3d with three axis, the third one using log"""
    X = df.drop(columns=["hazardous"])
    Y = df["hazardous"]

    limit = len(X)

    sns.reset_orig()

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X.loc[Y == True, x][:limit],
        X.loc[Y == True, y][:limit],
        X.loc[Y == True, z][:limit],
        c="blue",
        marker=".",
        s=10,
        label="hazardous",
    )
    ax.scatter(
        X.loc[Y == False, x][:limit],
        X.loc[Y == False, y][:limit],
        X.loc[Y == False, z][:limit],
        c="gray",
        marker=".",
        s=10,
        label="non hazardous",
        alpha=0.2,
    )

    ax.set_xlabel(x, size=16)
    ax.set_ylabel(y, size=16)
    ax.set_zlabel(z, size=16)
    ax.set_title("Hazardous and Non Hazardous objects in Earth's vicinity", size=20)

    plt.axis("tight")
    ax.grid(1)

    normalMarker = mlines.Line2D(
        [],
        [],
        linewidth=0,
        color="gray",
        marker=".",
        markersize=10,
        label="non hazardous",
    )

    anomalyMarker = mlines.Line2D(
        [], [], linewidth=0, color="blue", marker=".", markersize=10, label="hazardous"
    )

    plt.legend(
        handles=[normalMarker, anomalyMarker],
        bbox_to_anchor=(1.20, 0.38),
        frameon=False,
        prop={"size": 16},
    )
    plt.show()


show3D_data(df_stat, "est_diameter_max", "relative_velocity", "absolute_magnitude")


# Has can be seen on this first representation, the velocity seems to be slightly higher for objects considered a threat compared to the rest. The magnitude also seems to be capped at 20 for the hazardous objects, a high magnitude impliying a very dim object in the darkness of space.

# In[ ]:


show3D_data(df_stat, "est_diameter_max", "relative_velocity", "miss_distance")


# On this second graph we can further see that the velocity seems to be an important factor but on the other end the miss distance is not very representative. Indeed the miss distance is not that important considering objects could be shifted our of their orbit very easily by the cosmic billiard played in the solar system by gravity. An oibject that missed earth by a lot could still be a threat.