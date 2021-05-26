import numpy as np
import pandas as pd
import random

#Definition of the class Wrapper
#NEEDED INPUT: the corpus and the pre-trained Hate-Speech Classifier to get the BB predictions / predictions, i.e. the predict_proba function
#Class for realizing *hate.predict* agnostically

class Wrapper:
  def predict(self, corpus, predict_proba):
    output = []
    for item in corpus:
      output.append(predict_proba(item)) 
    return output
  
  def predict_labels(self, scores):
    labels=[]
    for score in scores:
      if score[0]>score[1]:
        labels.append(0)
      else:
        labels.append(1)
    return labels  

'''*Inspo from summary of CheckList*
In our case, we pass a corpus or a fraction of it (i.e. a collection of phrases). What if the sentences are too many or the users only wants to assess k phrases? 

Outcomes from options of customed specified k:

1.   Not specify anything, k would be 1 i.e. one random sentence from the corpus
2.   Specified k could be < len(corpus), i.e. only k random sentences are extracted 
3.   K could be = len(corpus), so in this case we loop through the entire corpus 

K for identifying a specific phrase is not needed, because the user could pass as a selection of corpus the phrase of interest in brackets, i.e. ['I hate you']; some for a collection of phrases of interest, the user passes it and then specifies as k the len of the collection
'''

class Evaluation:
  def __init__(self, real, predicted, proba, corpus, k=5):
    self.real=real
    self.predicted=predicted
    self.proba=proba
    self.corpus=corpus
    self.k=k # k is a parameter that specify how many records from the corpus analyse 

  def eval(self):
    n_tot = len(self.real)
    if self.k>n_tot or self.k<=0:
      return "Incorrect k"
    failed_ids = []
    correct_ids = []
    if self.k!=n_tot:
      index=random.sample(range(1,n_tot), self.k) # k positions at random 
    else: 
      index=range(n_tot)
    for i in index:  
      if self.real[i]!=self.predicted[i]:
        failed_ids.append(i)
      else:
        correct_ids.append(i)
    failed = len(failed_ids)
    failed_rate = 100*failed/n_tot
    evaluation = {
        'n_tot': n_tot, 
        'failed': failed, 
        'failed_rate': failed_rate, 
        'failed_ids': failed_ids, 
        'correct_ids': correct_ids
    }
    return evaluation

  def print_evaluation(self,eval):
    print("Summary:")
    keys = ['Total number of test records', 'Failed predictions', 'Failed prediction percentage', 'Failed IDs records', 'Correct IDs records']
    i=0
    for key in eval:
      print(keys[i], '->', eval[key])
      i+=1
  
  def failed_samples(self,eval):
    miscl_samples = []
    for item in eval['failed_ids']:
        miscl_samples.append(self.corpus[item])
    return miscl_samples

  def correct_samples(self,eval):
    wellcl_samples = []
    for item in eval['correct_ids']:
        wellcl_samples.append(self.corpus[item])
    return wellcl_samples
  
  def samples(self,eval):
    miscl_proba = []
    wellcl_proba = []
    miscl_label_real = []
    wellcl_label_real = []
    for item in eval['failed_ids']:
        miscl_proba.append(self.proba[item])
        miscl_label_real.append(self.real[item])
    for item in eval['correct_ids']:
        wellcl_proba.append(self.proba[item])
        wellcl_label_real.append(self.real[item])
    samples_dict = {
        'all_samples': self.failed_samples(eval)+self.correct_samples(eval),
        'all_proba': miscl_proba+wellcl_proba,
        'all_label_real': miscl_label_real+wellcl_label_real,
        'miscl_samples': self.failed_samples(eval), 
        'wellcl_samples': self.correct_samples(eval),
        'miscl_proba': miscl_proba, 
        'wellcl_proba': wellcl_proba,
        'miscl_label_real': miscl_label_real, 
        'wellcl_label_real': wellcl_label_real    
        }
    return samples_dict