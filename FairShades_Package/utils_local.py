import numpy as np
import pandas as pd
import pydotplus
from IPython.display import Image, display_svg, SVG
import matplotlib.pyplot as plt 
import seaborn as sns
import graphviz
from sklearn.tree import export_text

def heat_map(importances):
  print('Heatmap of the Feature Importances')
  sns.set()
  data_word = importances['feature']
  data_att=[]
  for item in importances['importance']:
    data_att.append(item)
  data_att=[data_att]
  d = pd.DataFrame(data=data_att, columns=data_word)
  f, ax = plt.subplots(figsize=(6,2))
  sns.heatmap(d, vmin=0, vmax=1.0, ax=ax, cmap="YlGnBu", yticklabels=['Importance'])
  label_x = ax.get_xticklabels()
  plt.setp(label_x, rotation=45, horizontalalignment='right')
  plt.show()
  print("Importance of:")
  print('-> Terms: ',data_word.tolist())
  print('-> Scores: ',data_att[0])

# from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def export_py_code(tree, feature_names, max_depth=100, spacing=4):
    if spacing < 2:
        raise ValueError('spacing must be > 1')
    # First: export tree to text
    res = export_text(tree, feature_names,#=features, 
                        max_depth=max_depth,
                        #decimals=3,
                        spacing=spacing-1)
    # Second: generate Python code from the text
    skip, dash = ' '*spacing, '-'*(spacing-1)
    code = 'Decision Tree Regressor on features: \n ({}):\n'.format(', '.join(feature_names))+'\n'
    #for line in repr(tree).split('\n'):
    #    code += skip + "# " + line + '\n'
    for line in res.split('\n'):
        line = line.rstrip().replace('|',' ')
        if '<' in line or '>' in line:
            line, val = line.rsplit(maxsplit=1)
            line = line.replace(' ' + dash, 'if')
            line = '{} {:g}:'.format(line, float(val))
        else:
            line = line.replace(' {} class:'.format(dash), 'return')
        code += skip + line + '\n'
    return code

'''
from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py   
'''

# computing the depth of each node and whether or not it is a leaf
def description(dt,d_keys):
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left #[i] id of the left child of node i or -1 if leaf node
    children_right = dt.tree_.children_right #[i]  id of the right child of node i or -1 if leaf node
    feature = dt.tree_.feature #[i] feature used for splitting node i
    threshold = dt.tree_.threshold #[i] threshold value at node i
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
    # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
    # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=d_keys[feature[i]],
                      threshold=round(threshold[i],2),
                      right=children_right[i]))

# rules used to predict a sample
def find_rules(dt,X_test,sample_id):
    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left #[i] id of the left child of node i or -1 if leaf node
    children_right = dt.tree_.children_right #[i]  id of the right child of node i or -1 if leaf node
    feature = dt.tree_.feature #[i] feature used for splitting node i
    threshold = dt.tree_.threshold #[i] threshold value at node i
    node_indicator = dt.decision_path(X_test)
    leaf_id = dt.apply(X_test)
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]
    print('Rules used to predict sample {id}:\n'.format(id=sample_id))
    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue
        # check if value of the split feature for sample 0 is below threshold
        if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        # print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
        print("decision node {node} : [Sample at index {sample}, term '{feature}'] the final score is = {value}) "
              "if the weight term is {inequality} {threshold})".format(
                  node=node_id,
                  sample=sample_id,
                  feature=d_keys[feature[node_id]],
                  value=X_test[sample_id, feature[node_id]],#round(X_test[sample_id, feature[node_id]],2),
                  inequality=threshold_sign,
                  threshold=round(threshold[node_id],2)))
        print()
        return X_test[sample_id, feature[node_id]]

# find -through decision paths comparison- which records on which leaves are finished wrt to the original
  # which is in the same leaf as x
def same_leave(sample_id, neigh,dt):
      leaves_ids = dt.tree_.apply(np.float32(neigh))
      leaves_id_base = leaves_ids[sample_id]
      common_leaves = np.where(leaves_ids==leaves_id_base)
      common_leaves = np.delete(common_leaves,np.where(common_leaves[0]==sample_id))
      print("In the neighbourhood of size "+str(len(leaves_ids))+", the original record"+#at index "+str(sample_id)+
            " is located in the leave id "+str(leaves_id_base)+". The records placed in the same leave are "+str(len(common_leaves))+" and are classified as Hateful with probability:")
      return leaves_id_base,common_leaves

def inv_samples(dt, original, neigh, BoW, sl, l_i, n=3): 
  DTR_pred = dt.tree_.value[l_i]
  BB_pred = []
  for index in sl:
    BB_pred.append(neigh[index-1][1][1])
  diff=[]
  invariant_s=[]
  for i in range(len(BB_pred)):
    diff.append(abs(BB_pred[i] - DTR_pred))
  index_invariant=sorted(range(len(diff)), key=lambda i: diff[i])[:n]
  for index in index_invariant:
    invariant_s.append(neigh[index-1])
  d = {
        'Leaf Hate value': round(DTR_pred[0][0],3),
        #'index_invariant':index_invariant,
        'Invariant samples and probabilities (non-hate, hate)':invariant_s
  }
  return d

'''
The ids from here are a little mixed: 
* in df and vectorized neigh. there is the original phrase starting from index 0 and the neigh. from index 1 
* in the structure returned by the neigh. generation process, there isn't: so from the start (index 0) there is the neigh. 
'''

# working with df['hate_proba']
def relevant_samples(original, neigh, n): 
    proba=original[1][1]
    diff=[]
    diff_with_signs=[]
    label_change=[]
    i_largest=[]
    temp=[]
    influential=[]
    not_influential=[]
    indexes_relevant=[]
    indexes_not_relevant=[]
    for i in range(len(neigh)):
      abs_diff=abs(proba - neigh[i][1][1])
      diff.append(abs_diff) # neigh does not contain the original record
      diff_with_signs.append(proba - neigh[i][1][1]) # removing abs because we need to keep the information about the increase or decrease
      if (proba < 0.5 and neigh[i][1][1] < 0.5):
        label_change.append('The label remains <non-hateful>.')
        not_influential.append(neigh[i]) 
        indexes_not_relevant.append(i)
      elif (proba < 0.5 and neigh[i][1][1] > 0.5):
        label_change.append('The label changes from <non-hateful> to <hateful>')
        i_largest.append(i)
        temp.append(abs_diff)
        #influential.append(neigh[i])
        #indexes_relevant.append(i)
      elif (proba > 0.5 and neigh[i][1][1] > 0.5):
        label_change.append('The label remains <hateful>.')   
        not_influential.append(neigh[i]) 
        indexes_not_relevant.append(i)
      elif (proba > 0.5 and neigh[i][1][1] < 0.5):
        label_change.append('The label changes from <hateful> to <non-hateful>')  
        i_largest.append(i)
        temp.append(abs_diff)
        #influential.append(neigh[i])
        #indexes_relevant.append(i)
    temp_indexes_relevant=sorted(range(len(temp)), key=lambda i: temp[i], reverse=True)[:n]
    for index in temp_indexes_relevant:
      indexes_relevant.append(i_largest[index])
    for index in indexes_relevant:
      influential.append(neigh[index])
    #i_largest=sorted(range(len(diff)), key=lambda i: diff[i], reverse=True)[:n]
    #i_smallest=sorted(range(len(diff)), key=lambda i: diff[i])[:n]
    #for index in i_largest:
    #  influential.append(neigh[index])
    #for index in i_smallest:
    #  not_influential.append(neigh[index])
    d = {
        'differences':diff,
        'differences_with_signs':diff_with_signs,
        'label_change':label_change,
        'indexes_relevant':indexes_relevant,#i_largest,
        'indexes_not_relevant':indexes_not_relevant[:n],#i_smallest,
        'influential_samples':influential, #controfattuali 
        'not_influential_samples':not_influential[:n] #prototipi 
    }
    return d

def find_different_terms(key,relevant,neigh):
  id_relevant=relevant[key] # could be influential or not, depending on key choice
  data=[]
  in_orig=[]
  relevant_changes=[]
  base=np.array(neigh[0]) #BoW representation of the original record
  for i in id_relevant:
    s=np.array(neigh[i+1]) 
    # we add one because the indexes start from the first neigh,
    # while the BoW representation include the original record, i.e. 
    # the first neigh. is in position 1, while it is indexed as 0 
    data.append(s)
  l=len(base[0]) # number of tokens in the original record
  for j in range(len(data)): # find for each phrase index of the terms that differs in the negh wrt to original
    temp_orig=[]
    temp_added=[]
    for i in range(l): # for each token in the orig. 
      if base[0][i]==1 and data[j][0][i]==0:
        temp_orig.append(i)
      elif base[0][i]==0 and data[j][0][i]==1:
        temp_added.append(i)
    in_orig.append(temp_orig)
    relevant_changes.append(temp_added)
  return in_orig,relevant_changes

# then ouput dict_keys 
def find_keys(a,dictionary):
  for i in range(len(a)): # for each phrase 
    for n in range(len(a[i])): # for each term (one or more)
      temp=a[i][n] #saving the index of the key 
      a[i][n]=dictionary[temp] # getting the value and replacing in the array a 
  return a 

# verbalize the explanation, i.e. contains (x addedd or substituted) ⇒ hate_proba goes up / down of y 
def pre_verbalize(key,terms_orig,terms_neigh,relevant,isLocal,tot=0,neigh=None,sentence_to_explain=None,real_label_sentence_to_explain=None,correct=None, isFirst=False):
  label_change=relevant['label_change']
  diff=relevant['differences']
  differences_with_signs=relevant['differences_with_signs']
  indexes=relevant[key]
  if key == 'indexes_relevant':
    samples_key='influential_samples'
  else:
    samples_key='not_influential_samples'
  result=[]
  Hate_increase=False
  for i in range(len(indexes)): # for each phrase 
    if differences_with_signs[indexes[i]] < 0: # retrieve the diff from the index 
      Hate_increase=True # proba was < proba_neigh, i.e. resulting in negative diff.
    result.append([relevant[samples_key][i][0],terms_neigh[i],terms_orig[i],diff[indexes[i]],label_change[indexes[i]],Hate_increase])
  if isLocal:
    # printing some captions 
    # tot=0
    if isFirst:
      for item in neigh[0].values():
        tot+=item
      print("The record you chose to explain is: ",sentence_to_explain)
      if real_label_sentence_to_explain == 1:
        print("It is an hateful record")
      else:
        print("It is a non-hateful record")
      if correct == False:
        print("The original record was wrongly classified by the Black Box! :( Don't worry, our Explainer will tell you more about it")
      else:
        print("The original record was correctly classified by the Black Box! :) Stay tuned for more details")
      tot=int (tot)
      print()
      print("-- > Total number of neighbours generated: ",tot)
      print("-- > Number of neighbours per capacity: ", neigh[0])
      d={}
      for key in neigh[0]:
        d[key]= round(100*neigh[0][key]/tot,2)
      print("-- > Percentage of neighbours per capacity ",d)
      print()
      print("------> Showing counterfactuals: ")
    else:
      print("------> Showing prototypes: ")
    for item in result:
      print('«',item[0],'»')
      if not item[1]: #addition 
        print('If <',item[2][0],'> is present, the difference in the probability w.r.t. <hateful> within the original record is of',round(item[3],2))
      else:
        print('If <',item[1][0],'> is present, the difference in the probability w.r.t. <hateful> within the original record is of',round(item[3],2))
      print(item[4]) 
      print()
  return tot,result 