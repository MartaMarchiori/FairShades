import numpy as np
import pandas as pd
import checklist
import checklist.editor
import checklist.text_generation
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.test_suite import TestSuite
from checklist.perturb import *
import nltk 
nltk.download('averaged_perceptron_tagger')
import spacy
import re
import random 

nlp = spacy.load('en_core_web_sm')
editor = checklist.editor.Editor()
editor.tg

############## terms addressing women  
women_noun = ['woman', 'sister', 'daughter', 'girl', 'wife', 'young lady', 'lady', 'bride',
              'grandmother', 'mum', 'housewife', 'girlfriend', 'partner']
women_noun_plural = ['women', 'sisters', 'aunts', 'daughters', 'girls', 'mothers', 'wives', 'young ladies', 'ladies',  
                     'grandmothers', 'mums', 'housewives', 'girlfriends', 'partners'] 

############## offensive 
offensive_women_noun = ['slut', 'bitch', 'whore', 'cunt', 'spinster', 'ditsy', 'hussy', 'frigid', 
                        'witch', 'wench', 'harpy', 'trollop', 'dyke', 'skank', 'butch']
offensive_women_noun_plural = ['sluts', 'bitches', 'whores', 'cunts']
offensive_generic = ['twat', 'prick', 'motherfucker', 'shit', 'shithead', 'dick', 'dickhead', 
                     'cock', 'cockface', 'cockhead', 'cocksucker', 'pussy', 'asshole', 'arsehole', 'assfucker', 
                     'fag', 'faggot', 'bastard', 'douche', 'bugger']
dis = ['weirdo', 'lame', 'cripple', 'invalid', 'inferm'] #'disabled', 'handicapped'
homeless = ['tramp', 'hobo', 'drifter', 'beggar', 'vagrant'] #'roofless', 'homeless', 'houseless'
old = ['codger', 'grandpa', 'oldie', 'geezer', 'elder']

############## work-related 
work_role = ['astronaut', 'academic', 'accountant', 'activist', 'administrator', 'advisor', 'agent', 'analyst','architect', 
             'assistant', 'attorney', 'artist' 'boxer', 'builder', 'businessman', 'cameraman', 'carpenter', 'clown', 
             'consultant', 'dentist', 'detective', 'developer', 'doctor', 'doorman', 'driver', 'electrician', 
             'economist', 'editor', 'educator', 'entrepreneur', 'executive',
             'engineer', 'farmer',  'fighter', 'fireman', 'footballer', 'foreman', 
             'historian', 'hitter', 'intern', 'interpreter', 'investigator', 'investor', 'gardener', 'housekeeper', 
             'journalist', 'librarian', 'lifeguard', 'magician', 'mechanic', 'nun', 'nurse', 'painter', 'paramedic', 
             'photographer', 'pilot', 'police', 'poet', 'postman', 'priest', 'professor', 'ranger', 'repairman', 'reporter', 
             'salesman', 'scientist', 'secretary', 'singer', 'soldier', 'student', 'surgeon', 'teacher', 'waiter', 'writer', 
             'attendant', 'officer', 'player', 'organizer', 'quarterback', 'shooter']
# these sterotyped lists are taken from the file male_occupations.txt and female_occupations.txt from the WiNo dataset 
fem_work_role = ['attendant', 'cashier', 'teacher', 'nurse', 'assistant', 'secretary', 'auditor', 'cleaner', 
                 'receptionist', 'clerk', 'counselor', 'designer', 'hairdresser', 'writer', 'housekeeper', 
                 'baker', 'accountant', 'editor', 'librarian', 'tailor']
male_work_role = ['driver', 'supervisor', 'janitor', 'cook', 'mover', 'laborer', 'construction worker',
                  'chief', 'developer', 'carpenter', 'manager', 'lawyer', 'farmer', 'salesperson', 'physician',
                  'guard', 'analyst', 'mechanic', 'sheriff', 'CEO']

############## other POS  
pos_adj = [                                                                             
    'adorable', 'amazing', 'astonishing', 'attractive', 'awesome', 'beautiful',
    'brilliant', 'caring', 'committed', 'cool', 'enthusiastic', 'excellent', 'exceptional',
    'extraordinary', 'fantastic', 'fun', 'good', 'great', 'happy', 'honest', 'impressive', 'incredible',
    'inspiring', 'intelligent', 'lovely', 'nice', 
    'successful', 'sweet', 'trustworthy', 'wise', 'wonderful']
neg_adj = [
    'absolute', 'aggressive', 'angry', 'annoying', 'arrogant', 
    'awful', 'bad', 'bitchy', 'bloody', 'boring', 'bossy', 'creepy', 'cruel', 
    'damn', 'dead', 'depressing', 'difficult', 'dirty', 'disappointing', 'disgusting', 
    'dreadful', 'emotional', 'evil', 'fake', 'fat', 'feisty', 'frustrating', 'fucking', 
    'giant', 'hormonal', 'horrible', 'hysterical', 'illogical', 'incompetent', 
    'irrational', 'irritating', 'lame', 'lousy', 'lying', 'nasty', 'poor', 
    'rough', 'sad', 'sassy', 'shameless', 'shitty', 'stupid', 'terrible', 'terrific', 
    'twatty', 'ugly', 'unhappy', 'wanky', 'weak', 'weird', 'whining', 'wicked', 'worst']
neutral_adj = [                                           
    'american', 'armenian','atheist', 'australian', 'black', 'british', 'christian', 'commercial', 
    'english', 'gay', 'immigrant', 'independent', 'independent', 'indian', 'indian', 'intellectual','busy', 
    'international', 'israeli', 'italian', 'jewish', 'married', 'muslim', 'old', 'poor',
    'private', 'rich', 'russian', 'single', 'tall', 'unmarried', 'white', 'working', 'young']
pos_verb_present = ['like', 'enjoy', 'appreciate', 'love', 'admire',
                   'respect', 'adore', 'support', 'care for', 'fancy', 'treasure', 'trust']
neg_verb_present = ['hate', 'dislike', 'regret', 'dread', 'despise', 'blame', 'hurt', 'envy', 'pity']
neutral_verb_present = ['see', 'find', 'miss', 'understand', 'believe', 'remember', 'talk to']
pos_verb_past = ['liked', 'enjoyed', 'appreciated', 'loved', 'admired', 
                 'respected', 'adored', 'supported', 'cared for', 'treasured', 'trusted']
neg_verb_past = ['hated', 'disliked', 'regretted', 'dreaded', 'despised','blamed', 'hurt', 'envied', 'pitied']
neutral_verb_past = ['saw', 'found', 'missed', 'understood', 'believed', 'remembered', 'talked to']
intens_adj = ['very', 'really', 'absolutely', 'truly', 'extremely', 'quite', 'incredibly', 'especially',
              'exceptionally', 'utterly', 'rather', 'totally', 'particularly',
              'remarkably', 'pretty', 'wonderfully', 'completely',
              'entirely', 'undeniably', 'highly']
intens_verb = ['really', 'absolutely', 'truly', 'extremely',  'especially',  'utterly',  'totally', 'particularly', 
               'highly', 'definitely', 'certainly', 'honestly', 'strongly', 'sincerely']
reducer_adj = ['somewhat', 'kinda', 'mostly', 'probably', 'generally', 'a little', 'a bit', 'slightly']

############## adding lexicons 
editor.add_lexicon('women_noun', women_noun, overwrite=True)
editor.add_lexicon('women_noun_plural', women_noun_plural, overwrite=True)
editor.add_lexicon('offensive_women_noun', offensive_women_noun, overwrite=True)
editor.add_lexicon('offensive_women_noun_plural', offensive_women_noun_plural, overwrite=True)
editor.add_lexicon('work_role', work_role, overwrite=True)
editor.add_lexicon('fem_work_role', fem_work_role, overwrite=True)
editor.add_lexicon('male_work_role', male_work_role, overwrite=True)
editor.add_lexicon('dis', dis, overwrite=True)
editor.add_lexicon('homeless', homeless, overwrite=True)
editor.add_lexicon('old', old, overwrite=True)
editor.add_lexicon('offensive_generic', offensive_generic, overwrite=True)
editor.add_lexicon('pos_verb_present', pos_verb_present, overwrite=True)
editor.add_lexicon('neg_verb_present', neg_verb_present, overwrite=True)
editor.add_lexicon('neutral_verb_present', neutral_verb_present, overwrite=True)
editor.add_lexicon('pos_verb_past', pos_verb_past, overwrite=True)
editor.add_lexicon('neg_verb_past', neg_verb_past, overwrite=True)
editor.add_lexicon('neutral_verb_past', neutral_verb_past, overwrite=True)
editor.add_lexicon('pos_verb', pos_verb_present+ pos_verb_past, overwrite=True)
editor.add_lexicon('neg_verb', neg_verb_present + neg_verb_past, overwrite=True)
editor.add_lexicon('neutral_verb', neutral_verb_present + neutral_verb_past, overwrite=True)
editor.add_lexicon('pos_adj', pos_adj, overwrite=True)
editor.add_lexicon('neg_adj', neg_adj, overwrite=True )
editor.add_lexicon('neutral_adj', neutral_adj, overwrite=True)
editor.add_lexicon('intens_adj', intens_adj, overwrite=True)
editor.add_lexicon('intens_verb', intens_verb, overwrite=True)
editor.add_lexicon('reducer_adj', reducer_adj, overwrite=True)

'''
**INPUT**: the type of capacity and perturbation; in case of Fairness capacity, the type of bias under investigation

**OUTPUT**: the neighborhood to be classified, divided wrt generation/perturbation typologies (i.e. type of capacity with different types of tests)

'''

class Neighbourhood(object):
  def generate_neighbourhood(self, capacity, test, sample):
    self.test = test
    self.sample = sample
    self.capacity = capacity
    method_name=capacity
    method=getattr(self,method_name,lambda :'Invalid')
    return method() 
  def fairness(self): 
    return self.empty(test_fairness(self.sample, self.test))
  def vocabulary(self):
    return self.empty(test_vocabulary(self.sample, self.test)) 
  def robustness(self):
    return self.empty(test_robustness(self.sample, self.test))
  def ner(self): 
    return self.empty(test_ner(self.sample, self.test))
  def taxonomy(self): 
    return self.empty(test_taxonomy(self.sample, self.test))

  def empty(self,ret):
    if len(ret[1])==0:
      print('No perturbation of the input could be applied') 
    else:
      return ret

  def auto(self):
    ret=[]
    neigh_type={}
    n_fair=0
    n_voc=0
    n_rob=0
    n_ner=0
    n_tax=0
    for item in self.sample:
      f=test_fairness([item], self.test)
      ret+=f[1]
      n_fair+=np.array(list(f[0].values())).flatten().sum()
      v=test_vocabulary([item], self.test)
      ret+=v[1]
      n_voc+=np.array(list(v[0].values())).flatten().sum() 
      r=test_robustness([item], self.test)
      ret+=r[1]
      n_rob+=np.array(list(r[0].values())).flatten().sum() 
      n=test_ner([item], self.test)
      ret+=n[1]
      n_ner+=np.array(list(n[0].values())).flatten().sum() 
      t=test_taxonomy([item], self.test)
      ret+=t[1]
      n_tax+=np.array(list(t[0].values())).flatten().sum() 
    neigh_type = {
        'fairness':n_fair,
        'vocabulary':n_voc,
        'robustness':n_rob,
        'ner':n_ner,
        'taxonomy':n_tax }
    print("Total number of neighbours generated: ",len(ret))
    return neigh_type,ret

# for wrong parameters combination
def check_param(d,test):
  if test in d.keys() or test == 'auto':
    return True 
  else:
    print('Specify a correct test name, choosing between')
    for key in d:
      print('- ',key)

# Append multiple value to a key in dictionary, from https://thispointer.com/python-dictionary-with-multiple-values-per-key/
def add_values_in_dict(sample_dict, key, list_of_values):
    if key not in sample_dict:
        sample_dict[key] = list()
    if list_of_values!= []:
      sample_dict[key].extend([len(list_of_values[0])])
    else:
      sample_dict[key].extend([0])
    return sample_dict


#### I) Fairness/Biases 

# from https://lgbta.wikia.org/wiki/Category:Sexuality, https://lgbta.wikia.org/wiki/Category:Gender
protected = { #'sexual': editor.template('{sexual_adj}').data,
    'sexuality': ['gay', 'lesbian', 'asexual', 'ace', 'bisexual', 'bi', 'homosexual', 'straight', 'cishet', 'heterosexual', 'pansexual', 'pan',
                  'demisexual', 'polysexual', 'bicurious', 'pancurious', 'polyamorous', 'aromantic', 'aro', 'biromantic', 'panromantic'], 
    'gender_identity': ['queer', 'trans', 'transgender', 'transsexual', 'cis', 'cisgender', 'cissexual', 'nonbinary', 'non-binary', 'enby', 'NB', 'genderqueer', 'genderfluid', 'genderflux', 'agender', 'bigender'],
    'race': ['black','hispanic', 'white', 'asian', 'european', 'latino', 'middle eastern', 'african', 'african american', 'american'],
    'religion': editor.template('{religion_adj}').data,
    'nationality': editor.template('{nationality}').data, 
    'country': editor.template('{country}').data,
    'city': editor.template('{city}').data,
    'male': editor.template('{male}').data,
    'female': editor.template('{female}').data,
    'last_name': editor.template('{last_name}').data,  
    'women_noun': editor.template('{women_noun}').data,
    'women_noun_plural': editor.template('{women_noun_plural}').data,
    'offensive_women_noun': editor.template('{offensive_women_noun}').data,
    'offensive_women_noun_plural': editor.template('{offensive_women_noun_plural}').data,
    'offensive_generic': editor.template('{offensive_generic}').data,
    'work_role': editor.template('{work_role}').data,
    'fem_work_role': editor.template('{fem_work_role}').data,
    'male_work_role': editor.template('{male_work_role}').data,
    'dis': editor.template('{dis}').data,
    'homeless': editor.template('{homeless}').data,
    'old': editor.template('{old}').data
}

def search_for_protected(x):
    found = []
    for sensitive in protected.keys():
      for p in protected[sensitive]:
          if re.search(r'\b%s\b' % p, x):
            found.append(sensitive)
    return found

def change_protected(x, meta=False, *args, **kwargs):
    ret = []
    ret_meta = []
    for p in protected[sensitive]:
        if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in protected[sensitive] if p != p2])
            ret_meta.extend([(p, p2) for p2 in protected[sensitive] if p != p2])
    if meta:
        return ret, ret_meta
    else:
        return ret

def change_stereotyped_work_roles(x, meta=False, *args, **kwargs):
    ret = []
    ret_meta = []
    for item in search_for_protected(x):
      if item=='fem_work_role':
        for p in fem_work_role:
            if re.search(r'\b%s\b' % p, x):
                ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in male_work_role if p != p2])
                ret_meta.extend([(p, p2) for p2 in male_work_role if p != p2])
      else:
        for p in male_work_role:
          if re.search(r'\b%s\b' % p, x):
            ret.extend([re.sub(r'\b%s\b' % p, p2, x) for p2 in fem_work_role if p != p2])
            ret_meta.extend([(p, p2) for p2 in fem_work_role if p != p2])
    if meta:
        return ret, ret_meta
    else:
        return ret

def test_fairness(samples, test):
  global sensitive
  neigh_type={}
  ret = []
  if check_param(protected,test):
    if test!='auto':
      sensitive=test
      if test!='stereotyped_work_roles':
        for item in samples:
          data = Perturb.perturb([item], change_protected, keep_original=False)
          ret += data.data
          neigh_type=add_values_in_dict(neigh_type, test, data.data)
      else:
        for item in samples:
          data = Perturb.perturb([item], change_stereotyped_work_roles, keep_original=False)
          ret += data.data
          neigh_type=add_values_in_dict(neigh_type, test, data.data)
    else:
      n=len(samples)
      for i in range(n):
        found = search_for_protected(samples[i].lower())
        if len(found)!=0:
          print('This is a list of protected entities present in your phrase: ',found)
          for identity in found:
            if identity not in neigh_type:
              neigh_type[identity]=[0] * n 
            else:
              pass
            ret_perturbed = []
            sensitive = identity
            ret_perturbed += Perturb.perturb([samples[i].lower()], change_protected, keep_original=False).data  
            neigh_type[identity][i]=len(ret_perturbed[0]) 
            temp=[]
            for phrase in ret_perturbed:
              temp+=phrase
            ret+=temp
  return neigh_type,ret 

#### II) Vocabulary 

neutral_words = set(
    ['.', 'the', 'The', ',', 'a', 'A', 'and', 'of', 'to', 'it', 'that', 'in',
     'this', 'for',  'you', 'there', 'or', 'an', 'by', 'about', 'my',
     'in', 'of', 'have', 'with', 'was', 'at', 'it', 'get', 'from', 'this'
    ])

forbidden = set(['No', 'no', 'Not', 'not', 'Nothing', 'nothing', 'without', 'but'] + pos_adj + neg_adj + pos_verb_present + pos_verb_past + neg_verb_present + neg_verb_past)

def change_neutral(d):
    examples = []
    subs = []
    words_in = [x for x in d.capitalize().split() if x in neutral_words]
    if not words_in:
        return None
    for w in words_in:
        suggestions = [x for x in editor.suggest_replace(d, w, beam_size=5, words_and_sentences=True) if x[0] not in forbidden]
        examples.extend([x[1] for x in suggestions])
        subs.extend(['%s -> %s' % (w, x[0]) for x in suggestions])
    if examples:
        idxs = np.random.choice(len(examples), min(len(examples), 10), replace=False)
        return [examples[i] for i in idxs]

def test_vocabulary(samples, test):
  ret=[]
  neigh_type={}
  for item in samples:
    d = {
      "neutral": Perturb.perturb([item], change_neutral, keep_original=False).data,
      }
    if check_param(d,test):
      if test!='auto':
        data=d[test]
        ret+=data
        neigh_type=add_values_in_dict(neigh_type, test, data)
      else:
        temp=[] # for each sample in the collection
        for key in d: # for each perturb
          data = d[key]
          for phrase in data: # for each phrase perturbed 
            temp+=phrase
          neigh_type = add_values_in_dict(neigh_type, key, data)
        if len(temp)!=0:
          ret+=temp
  return neigh_type,ret

#### III) Robustness 

import string 

def random_string(n):
    return ''.join(np.random.choice([x for x in string.ascii_letters + string.digits], n))
def random_url(n=6):
    return 'https://t.co/%s' % random_string(n)
def random_handle(n=6):
    return '@%s' % random_string(n)

def add_irrelevant(sentence):
    urls_and_handles = [random_url(n=3) for _ in range(5)] + [random_handle() for _ in range(5)]
    irrelevant_before = ['@miss '] + urls_and_handles
    irrelevant_after = urls_and_handles 
    rets = ['%s %s' % (x, sentence) for x in irrelevant_before ]
    rets += ['%s %s' % (sentence, x) for x in irrelevant_after]
    return rets

def add_emojis(sentence):
  perturb=[]
  for i in range(6):
    perturb.append(sentence+' '+random.sample(neutral_emojis,1)[0])
  return perturb

def add_hashtags(sentence):
  tagged = nltk.pos_tag(sentence.split())
  found = False
  for (word, tag) in tagged:
    if tag == 'JJ':
      perturb=sentence+' #'+word
      found=True
  if found:
    return perturb
  else:
    return 

emojis=pd.read_csv('https://raw.githubusercontent.com/abushoeb/EmoTag/master/data/EmoTag1200-scores.csv', header=0, error_bad_lines=False)

neutral_emojis=[]
for i in range(len(emojis)):
    if(emojis['anger'][i]<0.4 and emojis['disgust'][i]<0.4 and emojis['joy'][i]<0.4):
        neutral_emojis.append(emojis['emoji'][i])

def test_robustness(samples, test):
  ret=[]
  neigh_type={}
  for item in samples:
    d = {
      "irrelevant": Perturb.perturb([item], add_irrelevant, keep_original=False),
      "punctuation": Perturb.perturb(list(nlp.pipe([item])), Perturb.punctuation, keep_original=False),
      "contraction": Perturb.perturb([item], Perturb.contractions, keep_original=False), # ALSO IN TEXTATTACK
      "typos": Perturb.perturb([item], Perturb.add_typos, keep_original=False), 
      "emojis": Perturb.perturb([item], add_emojis, keep_original=False), 
      "hashtags": Perturb.perturb([item], add_hashtags, keep_original=False)
      }
    if check_param(d,test):
      if test!='auto':
        data=d[test].data
        ret+=d[test].data
        neigh_type=add_values_in_dict(neigh_type, test, data)
      else:
        temp=[] # for each sample in the collection
        for key in d: # for each perturb
          data = d[key].data
          for phrase in data: # for each phrase perturbed 
            temp+=phrase
          neigh_type = add_values_in_dict(neigh_type, key, data)
        if len(temp)!=0:
          ret+=temp
  return neigh_type,ret

#### IV) NER

def test_ner(samples, test):
  ret=[]
  neigh_type={}
  for item in samples:
    d = {
      "locations": Perturb.perturb(list(nlp.pipe([item])), Perturb.change_location, keep_original=False), 
      "numbers": Perturb.perturb(list(nlp.pipe([item])), Perturb.change_number, keep_original=False) 
      }
    if check_param(d,test):
      if test!='auto':
        data=d[test].data
        ret+=d[test].data
        neigh_type=add_values_in_dict(neigh_type, test, data)
      else:
        temp=[] # for each sample in the collection
        for key in d: # for each perturb
          data=d[key].data
          for phrase in data: # for each phrase perturbed 
            temp+=phrase
          neigh_type = add_values_in_dict(neigh_type, key, data)
        if len(temp)!=0:
          ret+=temp
  return neigh_type,ret

#### V) Taxonomy

#Using WordNet to automatically replace adjectives
def change_syn(x):
  syn = find_replacement_syn(x)
  if syn!=None: 
    ret = []
    ret.extend([re.sub(r'\b%s\b' % syn[0], syn[1], x)])  
    return ret 

def find_replacement_syn(s):
  tagged = nltk.pos_tag(s.split())
  for (word, tag) in tagged:
    if tag == 'JJ':
      if len(editor.synonyms(s, word))!=0:
        return word, editor.synonyms(s, word)[0]
      else: 
        return None

def test_taxonomy(samples, test):
  ret=[]
  neigh_type={}
  for item in samples:
    d = {
      "adj_synonyms": Perturb.perturb([item], change_syn, keep_original=False).data,
      }
    if check_param(d,test):
      if test!='auto':
        data=d[test]
        ret+=data
        neigh_type=add_values_in_dict(neigh_type, test, data)
      else:
        temp=[] # for each sample in the collection
        for key in d: # for each perturb
          data = d[key]
          for phrase in data: # for each phrase perturbed 
            temp+=phrase
          neigh_type = add_values_in_dict(neigh_type, key, data)
        if len(temp)!=0:
          ret+=temp
  return neigh_type,ret


