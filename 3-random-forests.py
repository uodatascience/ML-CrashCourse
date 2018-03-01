

# Using code from https://www.kaggle.com/depture/multiclass-and-multi-output-classification


# Random Forests with `sklearn`

## Random Forests
'''
We have already seen decision trees, but they tend to overfit if allowed to grow to their full depth.
Random forests use an ensemble of decision trees, each trained with an incomplete subset of the data,
to reduce variance at the (lower) expense of increased bias. Specifically, each tree receives a subset
of the training samples, and when each branch is split (an additional decision is added to the tree)
it uses a random subset of the sample features. The final class estimate is the average of predictions
across trees.

In this example, we will be classifying Myers-Briggs personality type from forum comments using this
dataset: https://www.kaggle.com/datasnaek/mbti-type . Some of this code was graciously borrowed from
depture's kernel here: https://www.kaggle.com/depture/multiclass-and-multi-output-classification

I have been warned to tell y'all that apparently personality psychologists view myers-briggs as snakeoil,
for us that doesn't matter so much, we're just trying to classify some stuff.
'''

# Cleaning Data
'''
This example requires several packages:
* `numpy`
* `scipy`
* `matplotlib`
* `nltk`
* `sklearn`
* `pandas`

If you don't have them installed, you will need to do so, presumably with `pip`. eg.

pip install numpy scipy
pip install sklearn

if you don't have `pip`, installation instructions can be found here:
https://pip.pypa.io/en/stable/installing/

If you don't have `python`, installation instructions can be found here:
https://wiki.python.org/moin/BeginnersGuide/Download

First we will import all our packages...
'''

import re
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, permutation_test_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

'''If this is the first time you are using `nltk`, you will need to download the stopwords and wordnet data'''

# nltk.download("stopwords")
# nltk.download("wordnet")

'''
Now we will need to load and clean the data. We will be turning the posts into a
list of lowercase words without punctuation, removing common words, and converting
the list of words to a "tf-idf," or term-frequency times inverse document-frequency,
matrix. The tf-idf representation is a *vectorization* of the text -- since a
numerical identity is just as useful to most learning algorithms as the character
representation of a word (if they can use them at all), we replace each unique word
with a number, or literally an index in a matrix. Since many words will be shared by
people of different classes (yno, the nature of language), tf-idf weighting emphasizes
the words that are unique to a particular sample. The term frequency for each sample
is multiplied by 1/the frequency of those terms for all samples. The equation used by
`sklearn`'s vectorizer is:

$$tfidf(t) = tf(t,d) * (log \frac{1+n_{d}}{1+df(d,t)} + 1) $$

where $tf(t,d)$ is the term frequency for term $t$ in document (sample) $d$, $n_{d}$
is the number of documents, and $df(d,t)$ is the number of samples that contain $t$.
'''

# obvs change this to wherever you have your stuff
data_path = '/Users/Jonny/Downloads/mbti_1.csv'
base_dir = '/Users/Jonny/rf/'

data = pd.read_csv(data_path)

# No idea why they ordered this this way...
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ',
                    'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ',
                    'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

# Make an encoder that will convert each of the personality types to a number
lab_encoder = LabelEncoder().fit(unique_type_list)

####
# We will use a lemmatizer to convert words with similar meanings to a common word,
# eg. am, are, is -> be
lemmatiser = WordNetLemmatizer()

# We will remove all "stopwords" or common words like "a, the, but" etc.
try:
    cachedStopWords = stopwords.words("english")
except:
    nltk.download('stopwords')
    cachedStopWords = stopwords.words("english")

# This function does some standard text tidying and returns two arrays of text samples and personality types
def pre_process_data(data, remove_stop_words=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts

        # Remove urls
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts)

        # Keep only words (no punctuation)
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove all >1 spaces
        temp = re.sub(' +', ' ', temp).lower()

        # Remove stop words and lemmatize
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)
        list_posts.append(temp)

    # del data
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality

list_posts, list_personality = pre_process_data(data, remove_stop_words=True)

# fkn save because good lord that takes awhile.
np.save("{}posts_preprocessed.npy".format(base_dir), list_posts)
np.save("{}types_preprocessed.npy".format(base_dir), list_personality)

# load em if we're starting in the middle
#list_posts = np.load("posts_preprocessed.npy")
#list_personality = np.load("types_preprocessed.npy")

# What words are common here?
words_tokenized = word_tokenize(" ".join(list_posts))
fdist = nltk.FreqDist(words_tokenized)

for word, frequency in fdist.most_common(20):
    print(u'{};{}'.format(word, frequency))

############
# That vectorizer we talked about. Exclude terms that are present in >90% and <1% of samples.
vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.01)

posts_tf = vectorizer.fit_transform(list_posts)

###########
# Split to test/train sets
X_train, X_test, Y_train, Y_test = train_test_split(posts_tf, list_personality, test_size=0.20)

np.save('{}X_train.npy'.format(base_dir), X_train)
np.save('{}X_test.npy'.format(base_dir), X_test)
np.save('{}Y_train.npy'.format(base_dir), Y_train)
np.save('{}Y_test.npy'.format(base_dir), Y_test)

#X_train = np.load('{}X_train.npy'.format(base_dir))
#X_test = np.load('{}X_test.npy'.format(base_dir))
#Y_train = np.load('{}Y_train.npy'.format(base_dir))
#Y_text = np.load('{}Y_test.npy'.format(base_dir))

'''
First just one forest. The only parameter we manipulate here is n_estimators,
or the number of trees in the forest. You can see the rest of the defaults with
help(RandomForestClassifier)

'''

# Create the classifier object
rfc_basic = RandomForestClassifier(n_estimators=100, n_jobs=6, verbose=1)

# Fit and score
rfc_basic.fit(X_train, Y_train)

# Accuracy...
rfc_basic.score(X_test, Y_test)

'''
Before we fine-tune our model, let's see if we need all these words...
We'll iteratively fit a bunch of trees using RFECV - recursive feature elimination
with cross validation. After each cross-validated set of trees is fit, the n least
informative features will be removed.
'''

# http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
# Select some features
rfc_elim = RandomForestClassifier(n_jobs=4, verbose=1,
                                  n_estimators=50)
rfc_rfecv = RFECV(rfc_elim, step=100, scoring="neg_log_loss", n_jobs=2, verbose=2)
rfc_rfecv.fit(X_train, Y_train)

joblib.dump(rfc_rfecv, '{}rfc_rfecv.pkl'.format(base_dir))
rfc_rfecv = joblib.load('{}rfc_rfecv.pkl'.format(base_dir))

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (neg log loss)")
plt.plot(range(1, len(rfc_rfecv.grid_scores_)*200 + 1, 200), rfc_rfecv.grid_scores_)
plt.show()

# subset optimal features
good_words = rfc_rfecv.get_support()
n_selected_words = np.sum(good_words)

X_test_sub = X_test[:,good_words]
X_train_sub = X_train[:,good_words]

np.save('{}X_train_sub.npy'.format(base_dir), X_train_sub)
np.save('{}X_test_sub.npy'.format(base_dir), X_test_sub)

#X_train_sub = np.load('{}X_train_sub.npy'.format(base_dir))
#X_test_sub = np.load('{}X_test_sub.npy'.format(base_dir))

# what are those words anyway?
tf_words = vectorizer.get_feature_names()
top_n_inds = np.where(good_words)[0]
top_n_words = [tf_words[i] for i in top_n_inds]

'''
Now we want to tune our hyperparameters. We will train a lot of models,
randomly varying in their parameters, and choose values that optimize time vs. model quality
'''

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
rfc_rand = RandomForestClassifier(verbose=1, n_jobs=2)
rfc_params = {"n_estimators": stats.randint(10, 150),
              "max_depth": [None, 3, 10, 20],
              "max_features": stats.randint(1, 100),
              "min_samples_split": stats.randint(2, 30),
              "min_samples_leaf": stats.randint(1, 10),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

n_rand_iter = 100
rand_search = RandomizedSearchCV(rfc_rand, param_distributions=rfc_params,
                                 n_iter=n_rand_iter, cv=3, verbose=2,
                                 scoring=["neg_log_loss", "f1_weighted", 'accuracy'],
                                 n_jobs=4, refit=False)

start = time()
rand_search.fit(X_train_sub, Y_train)
stop = time()
total_time = stop-start

joblib.dump(rand_search, '{}rand_search.pkl'.format(base_dir))
#rand_search = joblib.load('rand_search.pkl')

#rand_search.score(X_test_sub, Y_test)

rand_results = pd.DataFrame(rand_search.cv_results_)
rand_results['param_max_depth'][pd.isnull(rand_results['param_max_depth'])] = 100

scatter_vars = ['param_min_samples_leaf', 'param_n_estimators', 'param_min_samples_split', 'param_max_features', 'param_max_depth']

fig, ax = plt.subplots(3)
for v in scatter_vars:
    ax[0].scatter(rand_results['mean_test_neg_log_loss'], rand_results[v], label=v)
    ax[1].scatter(rand_results['mean_test_accuracy'], rand_results[v], label=v)
    ax[2].scatter(rand_results['mean_test_f1_weighted'], rand_results[v], label=v)
plt.legend()

bar_vars = ['param_bootstrap', 'param_criterion']
rand_results.boxplot(column="mean_test_neg_log_loss",by="param_bootstrap")
rand_results.boxplot(column="mean_test_neg_log_loss",by="param_criterion")


#############
'''
We can also use a grid of explicitly specified parameters...
'''

param_grid = {
    'n_estimators': [75,100,125],
    'max_features': [75, 100, 125]
}

rfc_search3 = RandomForestClassifier(n_jobs=2, verbose=1)

grid_search = GridSearchCV(rfc_search3, param_grid, n_jobs=4, verbose=2)
grid_search.fit(X_train_sub, Y_train)

rand_results3 = pd.DataFrame(grid_search.cv_results_)

scatter_vars = ['param_n_estimators', 'param_max_features']

fig, ax = plt.subplots()
for v in scatter_vars:
    ax.scatter(rand_results3['mean_test_score'], rand_results3[v], label=v)
plt.legend()

############
rfc_tuned = RandomForestClassifier(n_jobs=7, verbose=1, n_estimators=75,
                                   max_features=125, class_weight="balanced")
rfc_tuned.fit(X_train_sub, Y_train)

joblib.dump(rfc_tuned, '{}rfc_tuned.pkl'.format(base_dir))

rfc_tuned.score(X_test_sub, Y_test)

# Plot a confusion matrix
predictions = rfc_tuned.predict(X_test_sub)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predictions)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
ax.imshow(cm, interpolation='nearest')
plt.xticks(range(len(unique_type_list)), unique_type_list)
plt.yticks(range(len(unique_type_list)), unique_type_list)

# how does it compare to class frequency?
fig, ax = plt.subplots()
plt.hist(Y_train)
plt.xticks(range(len(unique_type_list)), unique_type_list)

# Which features (words) are important?
importances = rfc_tuned.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc_tuned.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1].tolist()

important_words = [top_n_words[i] for i in indices]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X_test_sub.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_test_sub.shape[1]), important_words)
#plt.xlim([-1, X.shape[1]])
plt.show()


##########
'''
The personality types aren't independent though, we actually have four pairs of two types.
We can explicitly model that by training a separate classifier for each pair.
'''

from sklearn.multioutput import MultiOutputClassifier

train_EI = np.zeros(len(Y_train), dtype=np.bool)
train_NS = np.zeros(len(Y_train), dtype=np.bool)
train_FT = np.zeros(len(Y_train), dtype=np.bool)
train_JP = np.zeros(len(Y_train), dtype=np.bool)

train_EI[np.isin(Y_train, [0,2,3,6,8,0,10,11])] = 1
train_NS[np.isin(Y_train, [8,9,10,11,12,13,14,15])] = 1
train_FT[np.isin(Y_train, [1,2,3,4,9,11,12,14])] = 1
train_JP[np.isin(Y_train, [1,2,6,7,8,9,12,13])] = 1

Y_train_multi = np.column_stack([train_EI, train_NS, train_FT, train_JP])

test_EI = np.zeros(len(Y_test), dtype=np.bool)
test_NS = np.zeros(len(Y_test), dtype=np.bool)
test_FT = np.zeros(len(Y_test), dtype=np.bool)
test_JP = np.zeros(len(Y_test), dtype=np.bool)

test_EI[np.isin(Y_test, [0,2,3,6,8,0,10,11])] = 1
test_NS[np.isin(Y_test, [8,9,10,11,12,13,14,15])] = 1
test_FT[np.isin(Y_test, [1,2,3,4,9,11,12,14])] = 1
test_JP[np.isin(Y_test, [1,2,6,7,8,9,12,13])] = 1

Y_test_multi = np.column_stack([test_EI, test_NS, test_FT, test_JP])


rfc_multi = RandomForestClassifier(n_estimators=100, max_features=100, class_weight="balanced", verbose=1, n_jobs=7)
rfc_multi_out = MultiOutputClassifier(rfc_multi)
rfc_multi_out.fit(X_train, Y_train_multi)

multi_predictions = rfc_multi_out.predict(X_test)

np.logical_and(multi_predictions, Y_test_multi)

#cm = confusion_matrix(Y_test_multi, multi_predictions)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#fig, ax = plt.subplots()
#ax.imshow(cm, interpolation='nearest')
#plt.xticks(range(len(unique_type_list)), unique_type_list)
#plt.yticks(range(len(unique_type_list)), unique_type_list)
