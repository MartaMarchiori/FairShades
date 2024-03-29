import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image, display_svg, SVG
import matplotlib.pyplot as plt 
import seaborn as sns
import graphviz
import nltk
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from utils_text_DTR import *
from utils_local import *
from utils_global import *
from utils_fairness_eval import *
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz import trees
from dtreeviz.utils import extract_params_from_pipeline
from dtreeviz.trees import *

# In the Explainer called function that performs the generation Extraction of the explanation, from text production of the derivative 
# Within a parameter of the explainer, personas, provide way at explanation time to change

class LocalExplanation(object): 
  def generate_l_explanation(self, input, tot=0, isLocal=True, neigh=None, sentence_to_explain=None, real_label_sentence_to_explain=None, correct=None):  
    self.input=input
    self.tot=tot
    self.isLocal=isLocal
    self.neigh=neigh
    self.sentence_to_explain=sentence_to_explain
    self.real_label_sentence_to_explain=real_label_sentence_to_explain
    self.correct=correct
    tweetTok = TweetTokenizer()
    orig_text=input[0][0] #passing the orig text
    orig_proba=[input[0][1][1]] #passing the prediction on the orig text
    if input[0][1][1] <= 0.5:
      self.isAbusive=False
    else: 
      self.isAbusive=True 
    counterfactuals=toDataFrame(input[1])
    self.size_local_neigh=len(input[1])
    self.df = pd.DataFrame() 
    self.df['text']=orig_text+counterfactuals[0] 
    self.df['abusive_proba']=orig_proba+counterfactuals[1] 
    df_not_tokenized=self.df.copy()
    self.df['text'] = tweets_preprocessing(self.df['text'])
    self.df['text']=self.df.text.apply((lambda x: tweetTok.tokenize(x)))
    self.neigh_BoW=vectorizer.fit_transform(self.df['text']).todense() 
    self.dt,self.d_keys,self.d_voc,importances=get_best_dtr(vectorizer,self.neigh_BoW,self.df['abusive_proba'],n=10)
    self.relevant_features=importances[importances['importance'] != 0] 
    id_relevant_features=self.relevant_features.index
    percentage = ( len(input[1]) * 10 ) / 100 
    top_k = round(percentage)
    self.relevant=relevant_samples(input[0], input[1], top_k)  

  def get_counterfactuals(self):
    type_of_samples='indexes_relevant'
    isFirst=True
    different_terms = find_different_terms(type_of_samples,self.relevant,self.neigh_BoW)
    terms_orig=find_keys(different_terms[0],self.d_keys)
    terms_neigh=find_keys(different_terms[1],self.d_keys)
    explanation=pre_verbalize(type_of_samples,terms_orig,terms_neigh,self.relevant,self.isLocal,self.tot,self.neigh,self.sentence_to_explain,self.real_label_sentence_to_explain,self.correct,isFirst)
    if self.isLocal==False:
      return explanation
  
  def get_prototypes(self):
    type_of_samples='indexes_not_relevant'
    isFirst=False
    different_terms = find_different_terms(type_of_samples,self.relevant,self.neigh_BoW)
    terms_orig=find_keys(different_terms[0],self.d_keys)
    terms_neigh=find_keys(different_terms[1],self.d_keys)
    explanation=pre_verbalize(type_of_samples,terms_orig,terms_neigh,self.relevant,self.isLocal,self.tot,self.neigh,self.sentence_to_explain,self.real_label_sentence_to_explain,self.correct,isFirst)
    if self.isLocal==False:
      return explanation
  
  def get_DTR_Is(self):
    C,P = get_I_from_DTR(self.dt,self.d_keys, self.isAbusive)
    if self.isLocal==True:
      print("------> Showing DTR counterfactuals:",C)
      print()
      print("------> Showing DTR prototypes:",P)
    if self.isLocal==False:
      return C,P,self.size_local_neigh

  def get_heatmap(self):
    print()
    heat_map(self.relevant_features)
  
  def get_all(self):
    self.get_counterfactuals()
    self.get_prototypes()
    self.get_DTR_Is()
    self.get_heatmap()


  def local_persona(self, persona_type, get): #default type: Data Scientist 
    self.get=get
    self.persona_type = persona_type
    method_name=persona_type
    method=getattr(self,method_name,lambda :'Invalid')
    return method() 


  def data_scientist(self): 
    if self.get=='DecisionTreeRegressor':
      self.DTR()
    elif self.get=='DTR properties':
      self.DTR_properties()

  def moderator_user(self): 
    if self.get=='DecisionTreeRegressor':
      self.DTR()

  def DTR(self):
    description(self.dt,self.d_keys)
    print()
    res = export_py_code(self.dt, self.d_keys, spacing=5)
    print(res)
  

  def DTR_properties(self):
    x_data=self.neigh_BoW
    x_data=np.array(x_data)
    features_model=self.d_keys
    print('Describing leaves')
    trees.viz_leaf_samples(self.dt, x_data, features_model, display_type='text')
    #trees.viz_leaf_samples(self.dt, x_data, features_model, figsize=(3,1.5))

    trees.viz_leaf_criterion(self.dt, display_type='text')
    #trees.viz_leaf_criterion(self.dt, display_type='plot', figsize=(3,1.5))

    print()  

    same_l=same_leave(0,self.neigh_BoW,self.dt)
    sl=same_l[1]
    l_i=same_l[0]
    orig=self.input[0]
    neigh=self.input[1]
    invariant_s=inv_samples(self.dt, orig, neigh, self.neigh_BoW, sl, l_i, 5)
    for key,value in invariant_s.items():
	    print('->',key, ':', value)

    scores_MAE = []
    for i in range(15):
      x_train, x_test, y_train, y_test = train_test_split(self.neigh_BoW, self.df['abusive_proba'], test_size= 0.15)
      s=local_fitting(x_train, x_test, y_train, y_test)
      scores_MAE.append(s)
    avg=average(scores_MAE)
    avg=round(avg,3)
    print()
    print('Evaluating the local fitting of the DTR w.r.t. the Black Box:')
    print('We test the DTR on 15 splits, randomly using each time 15% of the neighbourhood as test set')
    print('The average of the MAE scores obtained is = ',avg)
    print()

  def DTR_viz_tree(self):
    x_data=self.neigh_BoW
    x_data=np.array(x_data)
    features_model=self.d_keys
    dataset_target_reg=self.df['abusive_proba'] 
    target_reg='Abusive'
    sk_dtree = ShadowSKDTree(self.dt, x_data, dataset_target_reg, features_model, target_reg)
    viz = dtreeviz(self.dt,x_data,dataset_target_reg,target_name=target_reg,  # this name will be displayed at the leaf node
                  feature_names=features_model,title="Decision Tree Regressor on neighbourhood",title_fontsize=12,scale=2)
    return viz
  
  
  def DTR_viz_record(self):
    print("The path followed by the original record within the Decision Tree Regressor")
    x_data=self.neigh_BoW
    x_data=np.array(x_data)
    features_model=self.d_keys
    dataset_target_reg=self.df['abusive_proba'] 
    target_reg='Abusive'
    sk_dtree = ShadowSKDTree(self.dt, x_data, dataset_target_reg, features_model, target_reg)
    X = x_data[0]
    return trees.dtreeviz(sk_dtree, show_just_path=False, X = X, scale=1.5)

class GlobalExplanation(object):
####
  def generate_g_DTR_explanation(self, samples, corpus, inputs, bias, count_neigh_Fairness):  
      self.samples = samples
      self.corpus = corpus
      self.inputs = inputs
      self.bias = bias
      count_unfair_DTR=0
      res_C = []
      res_P = []
      n_dimensions=[]
      for i in range(len(inputs)): # calling local for each record 
        as_input = inputs[i]
        l_x=LocalExplanation()
        l_x.generate_l_explanation(as_input, len(as_input[1]), False)
        l_expl=l_x.get_DTR_Is()
        if l_expl[0] != []:
          res_C.append(l_expl[0]) # appending local c. to res 
          if unF_DTR(l_expl[0]):
              count_unfair_DTR+=1
        if l_expl[1] != []:
          res_P.append(l_expl[1]) # appending local p. to res
        n_dimensions.append(l_expl[2])
      len_dataset=len(inputs)
      mentions_C=[]
      for record in res_C:
          mentions_C+=record
      mentions_C=list(set(mentions_C))
      global protected_entities
      protected_entities = [] 
      for key in mentions_C:
        category = search_for_protected(key)
        if category:
          protected_entities.append([category[0], key])
      global sensitive_mentions
      sensitive_mentions = []
      for item in protected_entities:
        sensitive_mentions.append(item[1])
      #print("All the protected counterfactuals and their category: ",protected_entities)
      mentions_P=[]
      for record in res_P:
          mentions_P+=record
      mentions_P=list(set(mentions_P))
      mentions_P=clean_prototypes(mentions_P,mentions_C)
      global protected_entities_P
      protected_entities_P = [] 
      for key in mentions_P:
        category = search_for_protected(key)
        if category:
          protected_entities_P.append([category[0], key])
      global sensitive_mentions_P
      sensitive_mentions_P = []
      for item in protected_entities_P:
        sensitive_mentions_P.append(item[1])
      #print("All the protected prototypes and their category: ",protected_entities_P)
      fair=True 
      if count_unfair_DTR>0:
          fair=False 
          unfair_general =  count_unfair_DTR/len_dataset
          unfair_general = round(unfair_general,2)
      print("The records in the dataset are: ",len_dataset)
      print("For each record, the neighbourhoods generated are: ",n_dimensions)
      print("The records in the dataset that contain mentions to protected identities which cause discrimination are: ",count_unfair_DTR)
      print()
      print("(1) Is the BB fair, regarding",bias,"?",fair)
      print()
      if fair==False:
        print("(2) The BB is strictly UNFair regarding",bias,"at",unfair_general)
        print("    The UNFairness is computed as: N of records with at least one unfair neighbour (",count_unfair_DTR,") over N of records in the corpus (",len_dataset,")")
        print()
        print("(3) Counterfactual terms:")
        #print("All the counterfactuals: ",mentions_C)
        dic_C=grouping_DTR_influential(sensitive_mentions,protected_entities)
        for key,value in dic_C.items():
          print('    ', key, ':', value)
        #for i in range(len(protected_entities)):
        #  print('-> "',protected_entities[i][1],'" belongs to the protected attribute <',protected_entities[i][0],'>')
      #if fair==True: 
      #  print("All the counterfactuals: ",mentions_C)
      #print("All the prototypes: ",mentions_P)
      print()
      print("(4) Prototype terms:")
      dic_P=grouping_DTR_influential(sensitive_mentions_P,protected_entities_P)
      for key,value in dic_P.items():
          print('    ', key, ':', value)
      #for i in range(len(protected_entities_P)):
      #  print('-> "',protected_entities_P[i][1],'" belongs to the protected attribute <',protected_entities_P[i][0],'>')
####

  def generate_g_explanation(self, samples, corpus, inputs, bias, count_neigh_Fairness):  
    self.samples = samples
    self.corpus = corpus
    self.inputs = inputs
    self.bias = bias
    self.count_neigh_Fairness = count_neigh_Fairness
    res = []
    for i in range(len(inputs)): # calling local for each record 
      as_input = inputs[i]
      l_x=LocalExplanation()
      l_x.generate_l_explanation(as_input, len(as_input[1]), False)
      l_expl=l_x.get_counterfactuals()
      res.append(l_expl) # appending locals to res 
    len_dataset=len(inputs)
    scores_per_key, mentions = getting_mentions(res)
    global protected_entities
    protected_entities = [] 
    for key in mentions:
      category = search_for_protected(key)
      if category:
        protected_entities.append([category[0], key, mentions[key]])
    global sensitive_mentions
    sensitive_mentions = []
    for item in protected_entities:
      sensitive_mentions.append(item[1])
    count = 0 
    for item in samples['all_samples']:
      sensitive_present=False
      for word in sensitive_mentions: 
        if word in item: 
          sensitive_present=True 
      if sensitive_present==True:
        count+=1
    n_dimensions=[]
    for item in res: 
      n_dimensions.append(item[0])
    variations=average_sc(scores_per_key,sensitive_mentions)
    self.cf=inv_var_samples(res)
    count_neighs=self.cf['count']
    count_unfair=self.cf['count_unfair']
    count_unfair_per_neigh=self.cf['count_unfair_per_neigh']
    count_counterf_per_neigh=self.cf['count_counterf_per_neigh']
    count_relaxed_unfairness=0
    for i in range(len(count_unfair_per_neigh)):
      # c_un over counterf : x over 100 
      if count_counterf_per_neigh[i] != 0:
        if ((count_unfair_per_neigh[i]*100)/count_counterf_per_neigh[i]) > 5:
          count_relaxed_unfairness+=1
    fair = True 
    if self.cf['variant']:
      count_unfair_neigh,grouping_unfair = grouping_content_counterfactuals(self.cf['variant'],sensitive_mentions,protected_entities)
      if grouping_unfair:
        fair=False 
        unfair_general =  count_unfair/len_dataset
        unfair_general = round(unfair_general,2)
        unfair_relaxed =  count_relaxed_unfairness/len_dataset
        unfair_relaxed = round(unfair_relaxed,2)
    print("The records in the dataset are: ",len_dataset)
    print("For each record, the neighbourhoods generated are: ",n_dimensions)
    print("The records in the dataset that contain mentions to protected identities which cause discrimination are: ",count_unfair)
    print()
    print("------> Is the BB fair, regarding",bias,"?",fair)
    if fair==False:
      print("------> The BB is strictly UNFair regarding",bias,"at",unfair_general)
      print("------> The UNFairness is computed as: N of records with at least one unfair neighbour (",count_unfair,") over N of records in the corpus (",len_dataset,")")
      if unfair_relaxed != unfair_general:
        print()
        print("------> The BB is loosely UNFair regarding",bias,"at",unfair_relaxed)
        print("------> The UNFairness is computed as: N of records with = or > 5% of unfair neighbours (",count_relaxed_unfairness,") over N of records in the corpus (",len_dataset,")")
    print()
    if fair == False:
      for i in range(len(protected_entities)):
        print()
        print('-> If it is present the term <',protected_entities[i][1],'> that belongs to the protected group <',protected_entities[i][0],'> and that term is replaced by',protected_entities[i][2],':')
        if (variations[i][0]<0):
          print("On average, the Abusive probability increases compared to the original by",round(-variations[i][0],2),variations[i][1])
        else:
          print("On average, the Abusive probability decreases compared to the original by",round(variations[i][0],2), variations[i][1])
        print() 
    
  def get_prototypes(self):
    res = []
    for i in range(len(self.inputs)): # calling local for each record 
      as_input = self.inputs[i]
      l_x=LocalExplanation()
      l_x.generate_l_explanation(as_input, len(as_input[1]), False)
      l_expl=l_x.get_prototypes()
      res.append(l_expl)
    cf=inv_var_samples(res)
    if cf['invariant']:
      prot_words_sensitive=[]
      prot_words=[]
      for item in cf['invariant']:
        if item in protected_dictionaries_values:
          prot_words_sensitive.append(item[1])
        else:
          prot_words.append(item[1])
      if prot_words_sensitive:
        print('Sensitive "prototype" words that don\'t cause a flip in the label predicted: ')
        print(prot_words_sensitive)
      if prot_words:
        print('Other "prototype" words that don\'t cause a flip in the label predicted: ')
        print(prot_words)
    

  def global_persona(self, persona_type, get, path, data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1): #default type: Data Scientist 
    self.persona_type = persona_type
    self.get = get
    self.path = path
    self.data = data
    self.y_true = y_true
    self.y_pred = y_pred
    self.group_membership_data = group_membership_data
    self.memberships = memberships
    self.group_data = group_data
    self.protected_features = protected_features
    self.df1 = df1
    method_name=persona_type
    method=getattr(self,method_name,lambda :'Invalid')
    return method() 


  def data_scientist(self): 
    if self.get=='standard_metrics':
      textual_metrics(self.df1,self.y_true,self.y_pred)
    elif self.get=='fairness_metrics':
      Fairness_metrics(self.df1,self.y_true,self.y_pred)
    elif self.get=='all':
      textual_metrics(self.df1,self.y_true,self.y_pred)
      Fairness_metrics(self.df1,self.y_true,self.y_pred)


  def moderator_user(self): 
    if self.get=='fairness_metrics':
      Fairness_metrics(self.df1,self.y_true,self.y_pred)
    elif self.get=='all':
      Fairness_metrics(self.df1,self.y_true,self.y_pred)

  def domain_expert(self): 
    if self.get=='fairness_metrics':
      Fairness_metrics(self.df1,self.y_true,self.y_pred)
    elif self.get=='all':
      Fairness_metrics(self.df1,self.y_true,self.y_pred)


  def grouping_counterfactuals(self):
    c,grouping_unfair = grouping_content_counterfactuals(self.cf['variant'],sensitive_mentions,protected_entities)
    grouping_unfair_proba = grouping_probarange_counterfactuals(self.cf['variant'])
    if grouping_unfair:
      print("Grouping unfair counterfactuals w.r.t. terms: (category, term) -> phrases")
    if grouping_unfair_proba:
      for key,value in grouping_unfair.items():
        print(key, '->', value)
      print()
      print("Grouping unfair counterfactuals w.r.t. prediction range --> range: phrases")
      for item in grouping_unfair_proba:
        print(item)
