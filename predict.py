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

# In[1]:


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

# In[3]:


df = pd.read_csv("neo.csv")
# set the id column to be a category
df["id"] = df["id"].astype(object)
df["orbiting_body"] = df["orbiting_body"].astype("category")
df

# # Supervised Learning

# Now that the statistical analysis of the data is done, the next step is to try to predict the `hazardous` values from the rest of the data. Our first approach was with supervised learning. First, we clean the dataset to get quantitative data to determine wether or not the object is considerer a potential threat by the nasa, it is this column we will want to monitor. We remove constant values (sentry_object=False and orbiting_body=Earth).

# In[27]:


df_ml = df.drop(["name", "orbiting_body", "sentry_object"], axis=1)
df_ml = df_ml.set_index("id")
print(df_ml)


# ## Initialize train and test sets
# 
# To properly train our models on the data, we need to split the data in two parts, one for testing and one for training, this will allow proper scoring with data the models were not trained on.

# In[28]:


from sklearn.model_selection import train_test_split

X = df[
    [
        "est_diameter_max",
        "relative_velocity",
        "miss_distance",
        "absolute_magnitude",
    ]
]
y = df["hazardous"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# In[29]:


print(f"Number of training data: {len(y_train)}")
y_train.to_frame().hazardous.value_counts()


# In[30]:


(1 - 5891 / 54969) *100


# The data has been analyzed and cleaned; now, its time to build those machine learning models.
# 
# The train set contains 60860 data instances and has 5891 cases labelled as 1(hazardous), which means that if a model predicts all values as 0, then the accuracy will be 89.28%. This will be considered as baseline accuracy for the train set.
# Similarly, the baseline accuracy for the test set will be 89.08%.
# Our model should do better than these accuracies or should be robust enough to deal with the class imbalance.

# ## Let's try multiple models

# In[31]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
)
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def try_model(model):
    print(f"=============== {model} ===============")
    model.fit(X_train, y_train)

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    def print_metric(metric_name):
        print(f"------------ {metric_name} ------------")
        metric = metrics_dico[metric_name]
        print(f"train :\n{metric(y_train,y_train_predict)}")
        print(f"test :\n{metric(y_test,y_test_predict)}")

    metrics_dico = {
        "Classification Report": classification_report,
        "Accuracy Score": accuracy_score,
    }

    for metric_name in metrics_dico:
        print_metric(metric_name)

    conf_mat = confusion_matrix(y_test, y_test_predict, normalize='true')
    print(
        f"\n------------ Confusion matrix on test ------------\n{conf_mat}\n"
    )
    false_neg = conf_mat[1][0]
    false_pos = conf_mat[0][1]
    print(f"False negative : {false_neg}")
    print(f"False positive : {false_pos}")
    print(f"Test accuracy : {accuracy_score(y_test,y_test_predict)}")
    # plt.figure(figsize=(12, 12))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="g")
    plt.show()

# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data
pipe_tree = Pipeline(
    [
        # ('scale', StandardScaler()),
        ("pca", PCA(n_components=4)),
        ("tree", DecisionTreeClassifier()),
    ]
)

pipe_forest = Pipeline(
    [
        # ('scale', StandardScaler()), # scaling causes worse results
        ("pca", PCA(n_components=4)),
        ("tree", RandomForestClassifier(max_depth=3, random_state=0, n_estimators=50)),
    ]
)

models = [
    LogisticRegressionCV(class_weight="balanced"),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=0),
    pipe_tree,
    RandomForestClassifier(max_depth=3, random_state=0, n_estimators=50),
    pipe_forest
]

# for model in models:
#     try_model(model)


# Let's start with a logistic reression.

# In[32]:


try_model(LogisticRegressionCV())


# The confusion matrices clearly show that the model fails to predict even a single data instance as 1 (hazardous) and hence the model is not robust enough. 
# We can ask the model to balance the data :
# 
# The “balanced” mode of the *class_weight* parameter uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as $n\_samples / (n\_classes * np.bincount(y))$

# In[33]:


try_model(LogisticRegressionCV(class_weight='balanced'))


# We observe really bad performances : the model achieves 41.5% accuracy, way below the 89% baseline accuracy.
# 
# Let's try another strategy : K neighbors

# In[34]:


try_model(KNeighborsClassifier())


# If we look at the table for Naive Bayes, then we see that the accuracies for the test set and train set are equal to baseline accuracies. Moreover, if we look at the confusion matrix, we remark that the model almost always classifies as negative, meaning the model is broken and has predicted all values as 0 (not hazardous). Such a model is of zero significance because there’s no point in using a model when it can never fulfil its purpose.
# 
# The results we observe are surely caused by the fact that the dataset is un balanced. Let's use another strategy : Decision tree and ensembling.

# In[35]:


try_model(DecisionTreeClassifier(random_state=0))


# The accuracy for the test dataset remains equal to the baseline accuracy, but we can see that the accuracy of the train set is perfect ; the tree has overfit. To avoid that, we can choose a maximul depth.

# In[36]:


try_model(DecisionTreeClassifier(random_state=0, max_depth=3))


# We now have a test accuracy of 91, which is better than the baseline accuracy. An other way to avoid overfit is **bagging**.
# 
# ### Bagging
# 
# The principle of bagging is to choose **n** random subsets from the training set, and train **n** decision trees on it.
# For each candidate in the test set, Random Forest uses the class with the majority vote as this candidate’s final prediction.

# In[37]:


try_model(RandomForestClassifier(max_depth=3, random_state=0, n_estimators=50))


# It improves the result only by 0.003% . Let's try another strategy, **Boosting**
# 
# ### Boosting
# 
# Boosting model’s key is learning from the previous mistakes, e.g. misclassification data points. **n** estimators are trained sequentially ; they are trained with the residual errors of the previous tree, and the final prediction is made by simply adding up the predictions (of all trees).

# In[38]:


try_model(GradientBoostingClassifier(n_estimators=50))


# We improved the result by 0.7 %. We can improve the accuracy further by increasing the number of estimators :

# In[39]:


try_model(GradientBoostingClassifier(n_estimators=500))


# The accuracy improved by 0.42 %

# ### Dimension Reduction to facilitate classification

# We can try to reduce the dimension in order to accelerate the learning.

# In[40]:


from sklearn.svm import LinearSVC

pipeline = Pipeline(
    [
        # ('scale', StandardScaler()),
        # ("pca", PCA(n_components=3)),
        ("tree", LinearSVC(loss='hinge', dual=True, class_weight='balanced')),
    ]
)
try_model(pipeline)


# In[41]:


# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data
pipe_tree = Pipeline(
    [
        # ('scale', StandardScaler()),
        ("pca", PCA(n_components=3)),
        ("tree", DecisionTreeClassifier()),
    ]
)

pipe_forest = Pipeline(
    [
        # ('scale', StandardScaler()), # scaling causes worse results
        ("pca", PCA(n_components=3)),
        ("tree", RandomForestClassifier(max_depth=3, random_state=0, n_estimators=50)),
    ]
)

pipe_tree.fit(X_train, y_train)
y_pred_tree = pipe_tree.predict(X_test)

dec_tree_score_train = pipe_tree.score(X_train, y_train)
dec_tree_score_test = pipe_tree.score(X_test, y_test)
print(f"Score on training set : {dec_tree_score_train}")
print(f"Score on test set : {dec_tree_score_test}\n")


pipe_forest.fit(X_train, y_train)
y_pred_forest = pipe_forest.predict(X_test)

rand_forest_score_train = pipe_forest.score(X_train, y_train)
rand_forest_score_test = pipe_forest.score(X_test, y_test)
print(f"Score on training set : {rand_forest_score_train}")
print(f"Score on test set : {rand_forest_score_test}\n")


# In[42]:


conf_mat = confusion_matrix(y_test, y_pred_tree)
print(conf_mat)
false_neg = conf_mat[1][0]
false_pos = conf_mat[0][1]
print(f"False negative : {false_neg}")
print(f"False positive : {false_pos}")
# plt.figure(figsize=(12, 12))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="g")
plt.show()
pass


# In[43]:


conf_mat = confusion_matrix(y_test, y_pred_forest)
print(conf_mat)
false_neg = conf_mat[1][0]
false_pos = conf_mat[0][1]
print(f"False negative : {false_neg}")
print(f"False positive : {false_pos}")

# plt.figure(figsize=(12, 12))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="g")
plt.show()


# # Unsupervised

# In[44]:


from sklearn.model_selection import train_test_split

X = df[
    [
        "est_diameter_max",
        "relative_velocity",
        "miss_distance",
        "absolute_magnitude",
    ]
]


# In[45]:


from sklearn.cluster import KMeans
cs = []

ms = StandardScaler()

x = ms.fit_transform(X)

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()


# In[46]:


kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans.fit(X_train)

labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y_train == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))


# In[47]:


pipeline = Pipeline(
    [
        ('scale', StandardScaler()),
        # ('scale', MinMaxScaler()),
        # ("pca", PCA(n_components=3)),
        ("tree", KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)),
    ]
)
try_model(pipeline)


# # Conclusion
# 
# In the end we managed to try multiple models on our dataset, both supervised
# and unsupervised. In our case supervised learning seemed to be more appropriate
# and we decided to go further in this direction by trying more models. The
# unsupervised approach would probably require more data on each object to lead
# to a better detection of outliers. Another reason for the bad results of the
# unsupervised model may also be the unbaIn any case we managed to have a good
# accuracy with the supervised way. Finally our tentative to reduce the number of
# dimensions lead to acceptable results with a better speed.