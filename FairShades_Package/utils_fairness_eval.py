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
    #print('The *{}* group-based fairness metric for *{}* feature split '
    #      'are:'.format(metric_name, protected_feature))
    for grouping_i, grouping_name_i in enumerate(bin_names):
        j_offset = grouping_i + 1
        for grouping_j, grouping_name_j in enumerate(bin_names[j_offset:]):
            grouping_j += j_offset
            #is_not = ' >not<' if metric_matrix[grouping_i, grouping_j] else ''
            if metric_matrix[grouping_i, grouping_j]:
              print('The *{}* group-based fairness metric for *{}* feature split '
                    'are:'.format(metric_name, protected_feature))
              is_not = ' >not<'
              print('    * The fairness metric is{} satisfied for "{}" and "{}" '
                    'sub-populations.'.format(is_not, grouping_name_i,
                                              grouping_name_j))
              print('-------')
              #print()

# 1 
def textual_metrics(df,y_true,y_pred):
  grouped_metric = MetricFrame(skm.recall_score,
                              y_true, y_pred,
                              sensitive_features=df['group_membership_data'])
  #print("Overall recall = ", grouped_metric.overall)
  #print("Recall by groups = ", grouped_metric.by_group.to_dict())
  #print()
  #print("Min recall over groups = ", grouped_metric.group_min())
  #print("Max recall over groups = ", grouped_metric.group_max())
  #print("Difference in recall = ", grouped_metric.difference(method='between_groups'))
  #print("Ratio in recall = ", grouped_metric.ratio(method='between_groups'))
  #print()
  multi_metric = MetricFrame({'Precision':skm.precision_score, 'Recall':skm.recall_score},
                              y_true, y_pred,
                              sensitive_features=df['group_membership_data'], 
                              #sample_params = {'precision': {'zero_division': [0]}}
                            )
  print()
  print('Overall -->')
  print('Precision: ',multi_metric.overall[0])
  print('Recall: ',multi_metric.overall[1])
  print()
  print('By groups -->')
  print(multi_metric.by_group.head())

# 2
def Fairness_metrics(df,y_true,y_pred):
  dataset = np.array(df[['text','memberships']].to_records(index=False), dtype=[('text', '<U113'), ('memberships', '<U113')])
  # Select a protected feature
  protected_feature = 'memberships'
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

# 3 
def plots_metrics(df,y_true,y_pred, path):
  metrics = {
        'Accuracy': accuracy_score,
        'Precision': precision_score,
        'Recall': recall_score,
        'False Positive Rate': false_positive_rate,
        'True Positive Rate': true_positive_rate, 
        'Selection Rate': selection_rate,
        'Count': lambda y_true, y_pred: y_true.shape[0],
        }
  metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=df['memberships'])
  metric_frame.by_group.plot.bar(
        subplots=True, layout=[3,3], legend=False, figsize=[12,8],
        title='Show all metrics')
  plt.savefig(path+'/all_metrics.png', transparent=True)

# 4
def selection_rate_fbeta(df,y_true,y_pred):
  print()
  print('Overall -->')
  print("Selection Rate:", selection_rate(y_true, y_pred))
  print("Fbeta:", skm.fbeta_score(y_true, y_pred, beta=0.6))

  fbeta_06 = functools.partial(skm.fbeta_score, beta=0.6)

  metric_fns = {'Selection Rate': selection_rate, 'Fbeta': fbeta_06}

  grouped_on_memberships = MetricFrame(metric_fns,
                              y_true, y_pred,
                              sensitive_features=df['memberships'])
  print()
  print('By groups -->')
  print(grouped_on_memberships.by_group)

  disparity=df['memberships'].value_counts().index[0] # select the most frequent attribute
  df = df[df.memberships == disparity]
  y_true=df['y_true']
  y_pred=df['y_pred']
  df.head()
  grouped_on_disparity = MetricFrame(metric_fns,
                                y_true, y_pred,
                                sensitive_features=df['group_membership_data'])
  print()
  print('By group members -->')
  print(grouped_on_disparity.by_group.head())
