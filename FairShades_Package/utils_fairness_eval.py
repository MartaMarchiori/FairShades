import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image, display_svg, SVG
import matplotlib.pyplot as plt 
import seaborn as sns
import graphviz
import fairlearn
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.metrics import * 
import fatf.utils.models as fatf_models
import fatf.fairness.models.measures as fatf_mfm
import fatf.utils.metrics.tools as fatf_mt
import sklearn.metrics as skm
from sklearn.metrics import precision_score, recall_score, accuracy_score
import re
import functools
from neighbourhood_generation import protected

def add_values_in_dict_protected(sample_dict, key, list_of_values):
    if key not in sample_dict:
        sample_dict[key] = list()
    if list_of_values!= []:
      sample_dict[key].extend((list_of_values))
    else:
      sample_dict[key].extend([0])
    return sample_dict

# From https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_Wiki_Case_Study?hl=en and https://fairlearn.org/v0.6.1/user_guide/assessment.html#metrics and https://fat-forensics.org/sphinx_gallery_auto/fairness/xmpl_fairness_models_measure.html#sphx-glr-sphinx-gallery-auto-fairness-xmpl-fairness-models-measure-py
group_names = list(protected.keys())
num_groups = len(group_names)

def get_groups_memb(text):
    corpus=text.to_numpy()
    groups = [''] * len(corpus)
    for i in range(len(corpus)):
      for sensitive in protected.keys():
        for p in protected[sensitive]:
          if re.search(r'\b%s\b' % p, corpus[i]):
            groups[i]=p
    return groups

def get_groups_mod(text):
    # keys: protected x n of records
    # entries are protected values --> providing None/empty/'' 
    corpus=text.to_numpy()
    corpus_len=len(corpus)
    memberships = ['None'] * corpus_len
    groups = {}
    for k in group_names:
      groups[k]= ['None'] * corpus_len
    for i in range(corpus_len):
      for sensitive in group_names:
        for p in protected[sensitive]:
          if re.search(r'\b%s\b' % p, corpus[i]):
            groups[sensitive][i]=p
            memberships[i]=sensitive
    return memberships,groups

# > Fat - Forensics 
def print_fairness(metric_name, metric_matrix, protected_feature, bin_names):
    """Prints out which sub-populations violate a group fairness metric."""
    res=[]
    #print('The *{}* group-based fairness metric for *{}* feature split '
    #      'are:'.format(metric_name, protected_feature))
    for grouping_i, grouping_name_i in enumerate(bin_names):
        j_offset = grouping_i + 1
        for grouping_j, grouping_name_j in enumerate(bin_names[j_offset:]):
            grouping_j += j_offset
            #is_not = ' >not<' if metric_matrix[grouping_i, grouping_j] else ''
            if metric_matrix[grouping_i, grouping_j]:
              is_not = ' >not<'
              res.append([re.sub(r'[^\w]', '', grouping_name_i),re.sub(r'[^\w]', '', grouping_name_j)])
              #res.append([grouping_name_i,grouping_name_j])
              #print('    * The fairness metric is{} satisfied for "{}" and "{}" '
              #      'sub-populations.'.format(is_not, grouping_name_i,
              #                                grouping_name_j))
              #print('-------')
              #print()
    if res:
      print('The *{}* group-based fairness metric for *{}* feature split '
          'are:'.format(metric_name, protected_feature))
      print('    * The fairness metric is{} satisfied for sub-populations:'.format(is_not))
      keys=[]
      values=[]
      for item in res:
        keys.append(item[0])
      keys=list(set(keys))
      for k in keys:
        temp=[]
        for item in res:
          if(item[0]==k):
            temp.append(item[1])
        values.append(temp)
      d=dict(zip(keys, zip(values)))
      print(d)#res
      print('-------')


# 1 
def textual_metrics(df,y_true,y_pred):
  grouped_metric = MetricFrame(skm.recall_score,
                              y_true, y_pred,
                              sensitive_features=df['group_membership_data'])
  disparity=df['memberships'].value_counts().index[0] # select the most frequent attribute
  df = df[df.memberships == disparity]
  y_true=df['y_true']
  y_pred=df['y_pred']
  precision_metric = MetricFrame(skm.precision_score,
                              y_true, y_pred,
                              sensitive_features=df['group_membership_data'], 
                            )
  recall_metric = MetricFrame(skm.recall_score,
                              y_true, y_pred,
                              sensitive_features=df['group_membership_data'], 
                            )
  #accuracy_metric = MetricFrame(accuracy_score,
  #                            y_true, y_pred,
  #                            sensitive_features=df['group_membership_data'], 
  #                          )
  print()
  print('OVERALL -->')
  print('Precision: ',precision_metric.overall)
  print('Recall: ',recall_metric.overall)
  #print('Accuracy: ',accuracy_metric.overall)
  print()
  
  print('BY GROUP VALUES OF --> ',disparity)
  
  precision = pd.DataFrame(precision_metric.by_group)  
  precision = precision.rename(columns={"precision_score": 'Precision'})
  print(precision.sort_values(by=['Precision']).to_markdown())
  
  recall = pd.DataFrame(recall_metric.by_group)  
  recall = recall.rename(columns={"recall_score": 'Recall'})
  print(recall.sort_values(by=['Recall']).to_markdown())
  
  #accuracy = pd.DataFrame(accuracy_metric.by_group)  
  #accuracy = accuracy.rename(columns={"accuracy_score": 'Accuracy'})
  #print(accuracy.sort_values(by=['Accuracy']).to_markdown())
  

# 2
def Fairness_metrics(df,y_true,y_pred):
  disparity=df['memberships'].value_counts().index[0] # select the most frequent attribute
  df = df[df.memberships == disparity]
  y_true=df['y_true']
  y_pred=df['y_pred']
  dataset = np.array(df[['text','group_membership_data']].to_records(index=False), dtype=[('text', '<U113'), ('group_membership_data', '<U113')])
  # Select a protected feature
  protected_feature = 'group_membership_data'#'memberships'
  # Get a confusion matrix for all sub-groups according to the split feature
  confusion_matrix_per_bin, bin_names = fatf_mt.confusion_matrix_per_subgroup(
      dataset, 
      np.array(y_true), 
      np.array(y_pred), 
      protected_feature, 
      treat_as_categorical=True)
  print()
  # Get the Equal Accuracy binary matrix
  equal_accuracy_matrix = fatf_mfm.equal_accuracy(confusion_matrix_per_bin)
  print_fairness('Equal Accuracy', equal_accuracy_matrix, protected_feature, bin_names)
  print()
  # Get the Equal Opportunity binary matrix
  equal_opportunity_matrix = fatf_mfm.equal_opportunity(confusion_matrix_per_bin)
  print_fairness('Equal Opportunity', equal_opportunity_matrix, protected_feature, bin_names)
  print()
  # Get the Demographic Parity binary matrix
  demographic_parity_matrix = fatf_mfm.demographic_parity(confusion_matrix_per_bin)
  print_fairness('Demographic Parity', demographic_parity_matrix, protected_feature, bin_names)
  print()

