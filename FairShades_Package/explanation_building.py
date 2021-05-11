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

# In the Explainer called function that performs the generation Extraction of the explanation, from text production of the derivative ==> Within a parameter of the explainer, personas
# Provide way at explanation time to change

class LocalExplanation(object): 
  def generate_l_explanation(self, input, neigh, sentence_to_explain, real_label_sentence_to_explain, correct): #default type 
    self.input = input
    tweetTok = TweetTokenizer()
    orig_text=input[0][0] #passing the orig text
    orig_proba=[input[0][1][1]] #passing the prediction on the orig text
    counterfactuals=toDataFrame(input[1])
    global df
    df = pd.DataFrame() 
    df['text']=orig_text+counterfactuals[0] 
    df['hate_proba']=orig_proba+counterfactuals[1] 
    df_not_tokenized=df.copy()
    df['text'] = tweets_preprocessing(df['text'])
    df['text']=df.text.apply((lambda x: tweetTok.tokenize(x)))
    global neigh_BoW
    neigh_BoW=vectorizer.fit_transform(df['text']).todense() 
    global dt
    global d_keys
    global d_voc
    dt,d_keys,d_voc,importances=get_best_dtr(vectorizer,neigh_BoW,df['hate_proba'],n=10)
    relevant_features=importances[importances['importance'] != 0] 
    id_relevant_features=relevant_features.index
    percentage = ( len(input[1]) * 5 ) / 100 
    top_k = round(percentage)
    relevant=relevant_samples(input[0], input[1], top_k)
    different_terms = find_different_terms('indexes_relevant',relevant,neigh_BoW)
    terms_orig=find_keys(different_terms[0],d_keys)
    terms_neigh=find_keys(different_terms[1],d_keys)
    explanation=pre_verbalize('indexes_relevant',terms_orig,terms_neigh,relevant,neigh,sentence_to_explain,real_label_sentence_to_explain,correct)
    heat_map(relevant_features)
    return explanation     

  def generate_l_per_global(self, input, neigh): 
    self.input = input
    tweetTok = TweetTokenizer()
    orig_text=input[0][0] #passing the orig text
    orig_proba=[input[0][1][1]] #passing the prediction on the orig text
    counterfactuals=toDataFrame(input[1])
    df = pd.DataFrame() 
    df['text']=orig_text+counterfactuals[0] 
    df['hate_proba']=orig_proba+counterfactuals[1] 
    df_not_tokenized=df.copy()
    df['text'] = tweets_preprocessing(df['text'])
    df['text']=df.text.apply((lambda x: tweetTok.tokenize(x)))
    neigh_BoW=vectorizer.fit_transform(df['text']).todense() 
    dt,d_keys,d_voc,importances=get_best_dtr(vectorizer,neigh_BoW,df['hate_proba'],n=10)
    relevant_features=importances[importances['importance'] != 0] 
    id_relevant_features=relevant_features.index
    percentage = ( len(input[1]) * 5 ) / 100 
    top_k = round(percentage)
    relevant=relevant_samples(input[0], input[1], top_k)
    different_terms = find_different_terms('indexes_relevant',relevant,neigh_BoW)
    terms_orig=find_keys(different_terms[0],d_keys)
    terms_neigh=find_keys(different_terms[1],d_keys)
    explanation=local_per_global('indexes_relevant',terms_orig,terms_neigh,relevant,neigh)
    return explanation 


  def local_persona(self, persona_type='data_scientist'): #default type: Data Scientist 
    self.persona_type = persona_type

    method_name=persona_type
    method=getattr(self,method_name,lambda :'Invalid')
    return method() 


  def data_scientist(self): 
    self.DTR(dt)
    self.DTR_properties(dt,self.input)

  def moderator_user(self): 
    self.DTR(dt)


  def DTR(self, dt):
    description(dt,d_keys)
    print()
    res = export_py_code(dt, d_keys, spacing=5)
    print(res)
  

  def DTR_properties(self, tree_regressor, input):
    x_data=neigh_BoW
    x_data=np.array(x_data)
    features_model=d_keys
    print('Describing leaves (Bar charts at the bottom)')
    trees.viz_leaf_samples(tree_regressor, x_data, features_model, display_type='text')
    trees.viz_leaf_samples(tree_regressor, x_data, features_model, figsize=(3,1.5))

    trees.viz_leaf_criterion(tree_regressor, display_type='text')
    trees.viz_leaf_criterion(tree_regressor, display_type='plot', figsize=(3,1.5))

    print()  

    same_l=same_leave(0,neigh_BoW,dt)
    sl=same_l[1]
    l_i=same_l[0]
    orig=input[0]
    neigh=input[1]
    invariant_s=inv_samples(tree_regressor, orig, neigh, neigh_BoW, sl, l_i, 5)
    for key,value in invariant_s.items():
	    print('->',key, ':', value)

    scores_MAE = []
    for i in range(15):
      x_train, x_test, y_train, y_test = train_test_split(neigh_BoW, df['hate_proba'], test_size= 0.15)
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
    tree_regressor=dt 
    x_data=neigh_BoW
    x_data=np.array(x_data)
    features_model=d_keys
    dataset_target_reg=df['hate_proba'] 
    target_reg='Hate'
    sk_dtree = ShadowSKDTree(tree_regressor, x_data, dataset_target_reg, features_model, target_reg)
    viz = dtreeviz(tree_regressor,x_data,dataset_target_reg,target_name=target_reg,  # this name will be displayed at the leaf node
                  feature_names=features_model,title="Decision Tree Regressor on neighbourhood",title_fontsize=12,scale=2)
    return viz
  
  
  def DTR_viz_record(self):
    print("The path followed by the original record within the Decision Tree Regressor")
    tree_regressor=dt 
    x_data=neigh_BoW
    x_data=np.array(x_data)
    features_model=d_keys
    dataset_target_reg=df['hate_proba'] 
    target_reg='Hate'
    sk_dtree = ShadowSKDTree(tree_regressor, x_data, dataset_target_reg, features_model, target_reg)
    X = x_data[0]
    return trees.dtreeviz(sk_dtree, show_just_path=False, X = X, scale=1.5)

from raiwidgets import FairnessDashboard

class GlobalExplanation(object):
  def generate_g_explanation(self, samples, corpus, inputs):  
    self.samples = samples
    self.corpus = corpus
    self.inputs = inputs
    res = []
    for i in range(len(inputs)): # calling local for each record 
      global sentence_to_explain
      sentence_to_explain = corpus[i]
      l_x=LocalExplanation()
      l_expl=l_x.generate_l_per_global(inputs[i],len(inputs[i][1]))
      res.append(l_expl) # appending locals to res 
    len_dataset=len(inputs)
    scores_per_key, mentions = getting_mentions(res)
    global protected_entities
    protected_entities = [] 
    fair = True 
    for key in mentions:
      category = search_for_protected(key)
      if category:
        fair=False 
        protected_entities.append([category[0], key, mentions[key]])
    global sensitive_mentions
    sensitive_mentions = []
    for item in protected_entities:
      sensitive_mentions.append(item[1])
    count = 0 
    for item in samples['miscl_samples']:
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
    print()
    print("------> Is the BB fair?",fair,"<------")
    print()
    print("The records in the dataset are: ",len_dataset)
    print("For each record, the neighbourhoods generated are: ",n_dimensions)
    print("The records in the dataset that contain mentions to protected identities are: ",count)
    if fair == False:
      for i in range(len(protected_entities)):
        print()
        print('-> If it is present the term <',protected_entities[i][1],'> that belongs to the protected group <',protected_entities[i][0],'> and that term is replaced by',protected_entities[i][2],':')
        if (variations[i][0]<0):
          print("On average, the Hate probability increases compared to the original by",round(-variations[i][0],2),variations[i][1])
        else:
          print("On average, the Hate probability decreases compared to the original by",round(variations[i][0],2), variations[i][1])
    print()
    self.grouping(res)


  def global_persona(self, path, data,y_true,y_pred,group_membership_data,memberships,group_data,protected_features,df1, persona_type='data_scientist'): #default type: Data Scientist 
    self.persona_type = persona_type
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
    return method(path) 


  def data_scientist(self, path): # 1 2 3 4 + 5  
    textual_metrics(self.df1,self.y_true,self.y_pred)
    Fairness_metrics(self.df1,self.y_true,self.y_pred)
    plots_metrics(self.df1,self.y_true,self.y_pred,path)
    selection_rate_fbeta(self.df1,self.y_true,self.y_pred)

  def moderator_user(self, path): # 2 3 + 5 
    Fairness_metrics(self.df1,self.y_true,self.y_pred)
    plots_metrics(self.df1,self.y_true,self.y_pred, path)

  def domain_expert(self, path): # 2 3 + 5 
    Fairness_metrics(self.df1,self.y_true,self.y_pred)
    plots_metrics(self.df1,self.y_true,self.y_pred, path)


  def grouping(self, res):
    cf=inv_var_samples(res)
    grouping_unfair = grouping_content_counterfactuals(cf['variant'],sensitive_mentions,protected_entities)
    grouping_unfair_proba = grouping_probarange_counterfactuals(cf['variant'])
    print("Grouping unfair counterfactuals w.r.t. terms --> category, term: phrases")
    for key,value in grouping_unfair.items():
	    print(key, ':', value)
    print()
    print("Grouping unfair counterfactuals w.r.t. prediction range --> range: phrases")
    for item in grouping_unfair_proba:
      print(item)
