
import json
from agilkia import TraceSet
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json
import datetime
import copy

def load_from_dict(cls,data) -> 'TraceSet':
    """Load traces from the given file.
    This upgrades older trace sets to the current version if possible.
    """
    if isinstance(data, dict) and data.get("__class__", None) == "TraceSet":
        return cls.upgrade_json_data(data)
    else:
        raise Exception("unknown JSON file format: " + str(data)[0:60])

def load_traceset(datapath,dataset_name):


    with open(datapath+dataset_name+'.json') as json_file:
        data = json.load(json_file)

    traceset=load_from_dict(TraceSet,data)
    return traceset


def traceset_to_textset(traceset,start_end_token_creator=lambda i:('<sos>','<eos>'),format='str'):
    textset=[]
    for i,tr in enumerate(traceset):
        sos,eos=start_end_token_creator(i)
        if format=='str':
            textset.append(sos+' '+' '.join([ev.action for ev in tr.events])+' '+eos)
        elif format=='lst':
            textset.append(sos+[ev.action for ev in tr.events]+eos)
        else:
            raise ValueError('not implemented')
    return textset


def textset_to_bowarray(textset,vocabulary_provided=None):
    if vocabulary_provided is None:
        count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
        bowarray=count_vect_actions.fit_transform(textset)
        return bowarray.toarray(),count_vect_actions.vocabulary_
    else:
        count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
        count_vect_actions.vocabulary_=vocabulary_provided
        bowarray=count_vect_actions.transform(textset)
        return bowarray.toarray(),count_vect_actions.vocabulary_


def textset_to_one_hot(textset):
    count_vect_actions=CountVectorizer(stop_words=[],min_df=0.0,tokenizer=lambda x:x.split(' '),lowercase=False)
    bowarray=count_vect_actions.fit_transform(textset)
    one_hot=bowarray!=0
    return one_hot.toarray().astype(int),count_vect_actions.vocabulary_

def get_a_session(i,traceset_global):
    return [ev.action for ev in traceset_global[i].events]

if __name__=='__main__':
    datapath='./data/'
    dataset_names=["scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution","1026-steps"]
    dataset_id=3
    dataset_name=dataset_names[dataset_id]
    traceset=load_traceset(datapath,dataset_name)
    
    textset=traceset_to_textset(traceset)
    bowarray,voc=textset_to_bowarray(textset)
    print(bowarray)
    one_hot,voc=textset_to_one_hot(textset)
    print(one_hot)