import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image, display_svg, SVG
import matplotlib.pyplot as plt 
import seaborn as sns
import graphviz
import re 
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import statistics

def average(lst): 
  return statistics.mean(lst) 

def toDataFrame(classified_neigh):
  text=[]
  probas=[]
  for item in classified_neigh:
    text.append(item[0])
    probas.append(item[1][1]) # appending only Hateful confidence
  return text,probas

# https://medium.com/analytics-vidhya/pre-processing-tweets-for-sentiment-analysis-a74deda9993e
def tweets_preprocessing(tokenized):
  tok_tweets = tokenized.apply((lambda x: x.lower())) # lower
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r'{link}', '', x)) # placeholders 
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r"\[video\]", '', x)) # placeholders 
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r'&[a-z]+;', '', x)) #html
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r'@', '', x)) # mentions
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r'^RT[\s]+', '', x)) # retweets
  tok_tweets = tok_tweets.apply(lambda x: re.sub(r'^rt[\s]+', '', x))
  tok_tweets= tok_tweets.apply(lambda x: re.sub(r'\s+', ' ', x)) # substituting multiple spaces with single space
  return tok_tweets

'''https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/ 

(document_id, token_id nel dizionario-corpus) freq in BoW/score in TfIdf of a given token in a given document 

'''

#TF  = (Frequency of a word in the document)/(Total words in the document)
#IDF = Log((Total number of docs)/(Number of docs containing the word))

def foo(doc):
    return doc

vectorizer = CountVectorizer(
    analyzer='word',
    tokenizer=foo,
    preprocessor=foo,
    token_pattern=None, 
    #max_df=[0.85, 1.0], # This parameter is ignored if vocabulary is not None.
    binary=True # All non zero counts are set to 1
    )  

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=foo,
    preprocessor=foo,
    token_pattern=None, 
    binary=True,
    #smooth_idf=False, # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.
    #sublinear_tf=True # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    )

def dtree_grid_search(X,y,nfolds,n=10):
    param_grid = {'max_depth': np.arange(2, n), 'min_samples_split': np.arange(2, n), 'min_samples_leaf': np.arange(2, n)}
    dtree_model=DecisionTreeRegressor() # score function is R^2
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    dtree_gscv.fit(X, y)
    return dtree_gscv.best_params_

def get_best_dtr(vect_technique,neigh,proba,n=10):
  param=dtree_grid_search(neigh,proba,n)
  global dt 
  dt = DecisionTreeRegressor(max_depth=param['max_depth'], min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf'], random_state=0)
  dt.fit(neigh, proba)
  cv_results = cross_validate(dt, neigh, proba, cv=n, return_estimator=True)
  fi_list=[]
  for model in cv_results['estimator']:
    fi_list.append(model.feature_importances_)
  average_fi=np.mean(fi_list, axis=0)
  global d_keys
  d_keys=vect_technique.get_feature_names() #‘features’ holds a list of all the words in the tf-idf’s vocabulary, in the same order as the columns in the matrix
  global d_voc
  d_voc=vect_technique.vocabulary_
  global importances
  importances = pd.DataFrame({'feature':d_keys,'importance':np.round(average_fi,3)}) 
  importances = importances.sort_values('importance',ascending=False)
  return dt,d_keys,d_voc,importances

def local_fitting(x_train, x_test, y_train, y_test):
  clf = dt
  clf.fit(x_train, y_train)
  y_pred_DTR = clf.predict(x_test)
  maescore = mean_absolute_error(y_test, y_pred_DTR) 
  return maescore