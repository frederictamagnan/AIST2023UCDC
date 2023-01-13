from agilkia import TraceSet
from collections import Counter


def extract_all_n_grams(traceset,n=1):
    all_n_grams=[]
    traceset_list_format = []
    if isinstance(traceset, TraceSet):
        for tr in traceset:
            traceset_list_format.append([ev.action for ev in tr.events])

    elif isinstance(traceset, list):
        traceset_list_format=traceset
    else:
        raise ValueError('wrong traceset type, must be list of list or agilkia traceset')

    for tra in traceset_list_format:
        for i in range(0, len(tra) - n+1):
            actions = ",".join([ev for ev in tra[i:i + n]])
            all_n_grams.append(actions)
    return list(set(all_n_grams))

def extract_all_n_grams_weighted(traceset,n=1):
    traceset_list_format=[]
    if isinstance(traceset, TraceSet):
        for tr in traceset:
            traceset_list_format.append([ev.action for ev in tr.events])
    elif isinstance(traceset, list):
        traceset_list_format=traceset
    else:
        raise ValueError('wrong traceset type, must be list of list or agilkia traceset')

    all_n_grams=[]

    for tra in traceset_list_format:
        for i in range(0, len(tra) - n+1):
            actions = ",".join([ev for ev in tra[i:i + n]])
            
            all_n_grams.append(actions)
    counts=Counter(all_n_grams)
    return dict(counts)



def n_gram_coverage(candidate_traceset,reference_traceset,n=1):
    a=len(extract_all_n_grams(candidate_traceset,n=n))
    b=len(extract_all_n_grams(reference_traceset,n=n))
    return a/b

def n_gram_coverage_fixed(candidate_traceset,all_n_grams,n=1):
    return len(extract_all_n_grams(candidate_traceset, n=n)) / len(all_n_grams)