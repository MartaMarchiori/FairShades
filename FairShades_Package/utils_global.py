import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image, display_svg, SVG
import matplotlib.pyplot as plt 
import seaborn as sns
import graphviz
from neighbourhood_generation import *
from neighbourhood_structure import *
from utils_fairness_eval import *
import itertools

def add_mentions_in_dict_(sample_dict, key, list_of_values):
    if key not in sample_dict:
        sample_dict[key] = list()
    if list_of_values!= []:
      sample_dict[key].extend(list_of_values)
    return sample_dict

def average_sc(scores,sensitive_mentions):
  averages=[]
  for key in scores:
    if key in sensitive_mentions:
      sum=0
      score=scores[key]
      for item in score:
        if item[1]==True:
          value=-item[0]
        else:
          value=item[0]
        sum+=value 
      avg=sum/len(score)
      averages.append([avg, item[2]])
  return averages 

def getting_mentions(res):
  mentions = {}
  global scores_per_key 
  scores_per_key = {}
  for record in res:
    for sample in record[1]:
      if len(sample[2])==1:
        mentions=add_mentions_in_dict_(mentions, sample[2][0], sample[1])
        scores_per_key=add_mentions_in_dict_(scores_per_key, sample[2][0], [[sample[3],sample[5],sample[4]]])
      else:
        for key in sample[2]:
          mentions=add_mentions_in_dict_(mentions, key, sample[1])
          scores_per_key=add_mentions_in_dict_(scores_per_key, key, [[sample[3],sample[5],sample[4]]])
  return scores_per_key,mentions

def grouping_content_counterfactuals(counterfactuals,sensitive_mentions,protected_entities):
  count_unfair=0
  group = {}
  for item in counterfactuals:
    if len(sensitive_mentions)==1:
      if item[2] == sensitive_mentions:
        key=protected_entities[0][0],item[2][0]
        group=add_mentions_in_dict_(group, key, [[item[0],item[4]]])        
    else:
      for i in range(len(sensitive_mentions)):
        if item[2] == [sensitive_mentions[i]]:
          key=protected_entities[i][0],item[2][0]
          group=add_mentions_in_dict_(group, key, [[item[0],item[4]]])
  for k in group:
    count_unfair+=len(group[k])
  return count_unfair,group

def grouping_probarange_counterfactuals(counterfactuals):
  proba = []
  for item in counterfactuals:
    proba.append(item[3])
  proba.sort()
  groups = []
  for k, g in itertools.groupby(proba, key=lambda n: n//0.35):
    groups.append(list(g))
  group=[]
  for g in groups:
    temp=[]
    key='Prediction range from '+str (round(min(g),2))+' to '+str (round(max(g),2))
    temp.append(key)
    for item in counterfactuals:
      if item[3] in g: 
        temp.append(item[0])
    group.append(temp)
  return group

def inv_var_samples(res):
  variant=[] 
  invariant=[]
  count=0
  for item in res: 
    count+=item[0]
    for neigh in item[1]:
      if neigh[4]=='The label changes from <non-hateful> to <hateful>' or neigh[4]=='The label changes from <hateful> to <non-hateful>':
        variant.append(neigh)
      else:
        invariant.append(neigh)
  d = {
        'count':count,
        'variant':variant,
        'invariant':invariant, 
  }
  return d

def build_global_inputs(corpus,samples,predict_proba):
  global rlc
  rlc=[]
  inputs=[]
  for item in corpus:
    id = samples['all_samples'].index(item)
    real_label_sentence_to_explain=samples['all_label_real'][id]
    rlc.append(real_label_sentence_to_explain) #for Fairness Eval.
    n=Neighbourhood()
    neigh=n.generate_neighbourhood('auto', 'auto', [item], True)
    predictions = building_predicted_for_neigh(neigh,predict_proba)
    i=ClassificationInput()
    input=i.generate_input([samples['all_samples'][id]], samples['all_proba'][id], neigh[1], predictions)
    inputs.append(input)
  return inputs

def build_global_df(inputs):
  sum = []
  predicted = []
  text_corpus = []
  for item in inputs:
    if item[0][1][1]<0.5: # orig. 
      predicted.append(0) # non-hateful 
    else: 
      predicted.append(1) # hateful 
    text_corpus+=item[0][0]
    sum.append(1+len(item[1]))
    for n in item[1]: # neigh. 
      text_corpus.append(n[0])
      if n[1][1]<0.5:
        predicted.append(0) # non-hateful 
      else: 
        predicted.append(1) # hateful 
  real_labels=[]
  for i in range(len(sum)):
    for k in range(sum[i]):
      real_labels.append(rlc[i])
  data = pd.DataFrame(text_corpus) 
  data = data[0]
  y_true=real_labels 
  y_pred=predicted 
  group_membership_data = get_groups_memb(data) 
  temp = get_groups_mod(data)
  memberships = temp[0]
  group_data = temp[1]
  df1 = pd.DataFrame({
                'text': data,
                'y_true': y_true,
                'y_pred': y_pred, 
                'memberships': memberships,
                'group_membership_data': group_membership_data})
  protected_features = pd.DataFrame(group_data) # columns: protected x n of records; entries are protected values --> providing None/empty/'' 
  protected_features = protected_features.drop(index=df1.index[df1['memberships'] == 'None'].tolist())
  df1 = df1[df1.memberships != 'None']
  y_true=df1['y_true']
  y_pred=df1['y_pred'] 
  return data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1

def create_bias_keys():
  # SEXISM => Misogyny, gender and sexual orientation
  sexism_keys=['sexuality', 'gender_identity', 'women_noun', 'women_noun_plural', 'offensive_women_noun', 'offensive_women_noun_plural']#, 'fem_work_role', 'male_work_role']
  sexism = []
  for key in sexism_keys:
    sexism+=protected.get(key)
  # RACISM => Race, nationality and religion
  racism_keys=['race', 'religion', 'nationality', 'country', 'city']
  racism = []
  for key in racism_keys:
    racism+=protected.get(key)
  # ABLEISM => Disability
  ableism_keys=['dis', 'homeless', 'old']
  ableism = []
  for key in ableism_keys:
    ableism+=protected.get(key)
  return sexism,racism,ableism

def grouping_global_samples(x,k):
    for sensitive in k:
      if re.search(r'\b%s\b' % sensitive, x):
        return True

def divide_corpus_per_bias(corpus):
  sexism,racism,ableism=create_bias_keys()
  sexism_samples=[]
  racism_samples=[]
  ableism_samples=[]
  for item in corpus:
    if grouping_global_samples(item,sexism):
      sexism_samples+=[item]
    elif grouping_global_samples(item,racism):
      racism_samples+=[item]
    elif grouping_global_samples(item,ableism):
      ableism_samples+=[item]
  return sexism_samples,racism_samples,ableism_samples