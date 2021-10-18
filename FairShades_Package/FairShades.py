import numpy as np
import pandas as pd
import sys, os
from input_eval import *
from input_eval import Wrapper
from neighbourhood_generation import *
from neighbourhood_structure import *
from utils_text_DTR import *
from utils_local import *
from utils_global import *
from utils_fairness_eval import *
from explanation_building import *

class FairShades(object): 
  def toExplain(self, dataset, subset_records, predict_proba): 
    self.dataset = dataset
    self.subset_records = subset_records 
    self.predict_proba=predict_proba
    real_labels=dataset[:subset_records][1]
    w1 = Wrapper()
    pred_proba=w1.predict(dataset[:subset_records][0], predict_proba)
    pred_labels=w1.predict_labels(pred_proba)
    ev = Evaluation(real_labels, pred_labels, pred_proba, dataset[:subset_records][0], subset_records) 
    result = ev.eval()
    self.samples=ev.samples(result)
    ev.print_evaluation(result)
    return self.samples
  
  def neighbourhoodPrediction(self, sentence_to_explain, correct):
    self.sentence_to_explain=sentence_to_explain
    if correct==True:
      key_sample='wellcl_'
      index_sentence_to_explain = self.samples['wellcl_samples'].index(sentence_to_explain) 
    else:
      key_sample='miscl_'
      index_sentence_to_explain = self.samples['miscl_samples'].index(sentence_to_explain) 
    self.real_label_sentence_to_explain=self.samples[key_sample+'label_real'][index_sentence_to_explain]
    n=Neighbourhood()
    self.neigh=n.generate_neighbourhood('auto', 'auto', [sentence_to_explain])
    predictions = building_predicted_for_neigh(self.neigh,self.predict_proba)
    i=ClassificationInput()
    self.in_input=i.generate_input([self.samples[key_sample+'samples'][index_sentence_to_explain]], self.samples[key_sample+'proba'][index_sentence_to_explain], self.neigh[1], predictions)

  def explain_local(self, sentence_to_explain, correct, isAbusive):
    self.sentence_to_explain=sentence_to_explain
    self.correct=correct
    self.isAbusive=isAbusive
    self.neighbourhoodPrediction(sentence_to_explain,correct)
    l_x=LocalExplanation()
    l_expl=l_x.generate_l_explanation(self.in_input, 0, True, self.neigh, self.sentence_to_explain, self.real_label_sentence_to_explain, correct, isAbusive)
    return l_x
    
  def explain_global(self, corpus, bias):
    nFair,inputs=build_global_inputs(corpus, self.samples, self.predict_proba)
    data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1=build_global_df(inputs)
    g_x=GlobalExplanation()
    g_expl=g_x.generate_g_explanation(self.samples, corpus, inputs, bias, nFair)
    return g_x,data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1

  def explain_global_via_DTR(self, corpus, bias):
    nFair,inputs=build_global_inputs(corpus, self.samples, self.predict_proba)
    data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1=build_global_df(inputs)
    g_x=GlobalExplanation()
    g_expl=g_x.generate_g_DTR_explanation(self.samples, corpus, inputs, bias, nFair)
    return g_x,data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1