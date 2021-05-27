import numpy as np
import pandas as pd
import checklist
import checklist.editor
import spacy

nlp = spacy.load('en_core_web_sm')
editor = checklist.editor.Editor()
editor.tg

############## terms addressing women  
women_noun = ['woman', 'sister', 'daughter', 'girl', 'wife', 'young lady', 'lady', 'bride',
              'grandmother', 'mum', 'mother', 'housewife', 'girlfriend', 'partner']
women_noun_plural = ['women', 'sisters', 'aunts', 'daughters', 'girls', 'mothers', 'wives', 'young ladies', 'ladies',  
                     'grandmothers', 'mums', 'mothers', 'housewives', 'girlfriends', 'partners'] 

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


nationalities=[]
for item in editor.template('{nationality}').data:
  nationalities.append(item+'s') 

# from https://lgbta.wikia.org/wiki/Category:Sexuality, https://lgbta.wikia.org/wiki/Category:Gender
protected = { #'sexual': editor.template('{sexual_adj}').data,
    'sexuality': ['gay', 'lesbian', 'asexual', 'ace', 'bisexual', 'bi', 'homosexual', 'straight', 'cishet', 'heterosexual', 'pansexual', 'pan',
                  'demisexual', 'polysexual', 'bicurious', 'pancurious', 'polyamorous', 'aromantic', 'aro', 'biromantic', 'panromantic'], 
    'gender_identity': ['queer', 'trans', 'transgender', 'transsexual', 'cis', 'cisgender', 'cissexual', 'nonbinary', 'non-binary', 'enby', 'NB', 'genderqueer', 'genderfluid', 'genderflux', 'agender', 'bigender'],
    'race': ['black','hispanic', 'white', 'asian', 'european', 'latino', 'middle eastern', 'african', 'african american', 'american'],
    'religion': editor.template('{religion_adj}').data,
    'nationality': editor.template('{nationality}').data, 
    'nationalities': nationalities, 
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

