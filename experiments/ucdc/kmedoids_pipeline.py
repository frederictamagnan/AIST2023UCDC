from .utils.clustering_pipeline import ClusteringPipeline
from .utils.sample_heuristic import SampleHeuristic
from .utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from sklearn.preprocessing import StandardScaler
# from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.kmedoids import kmedoids
from collections import defaultdict
from random import sample
import gensim
import numpy as np
from math import sqrt

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

class KMedoidsPipeline(ClusteringPipeline):
    
    def __init__(self):
        pass
    def preprocessor(self, execution_traces_agilkia_format):
        textset=traceset_to_textset(execution_traces_agilkia_format)
        lang=Lang("traces")
        for sentence in textset:
            lang.addSentence(sentence)
        
        listset=[indexesFromSentence(lang,sentence) for sentence in textset]
        X=np.zeros((len(textset),len(textset)))
        for i,sentence1 in enumerate(listset):
            for j,sentence2 in enumerate(listset):
                X[i,j]=self.similarity(sentence1,sentence2)
            if i % 10==0:
                print(i)
        return X
                
        
        
    def fit_predict(self, preprocessed_execution_traces,k):
        initial_medoids=list(range(0,k))
        # model=KMedoids(n_clusters=k,metric='precomputed', init='k-medoids++')
        # model.fit(preprocessed_execution_traces)
        # return model.fit_predict(preprocessed_execution_traces)
        kmedoids_instance=kmedoids(preprocessed_execution_traces,initial_medoids,data_type='distance_matrix')
        kmedoids_instance.process()
        clusters_list=kmedoids_instance.get_clusters()
        
        labels=[0 for i in range(len(preprocessed_execution_traces))]
        for k,cluster in enumerate(clusters_list):
            for idx in cluster:
                labels[idx]=k
        return labels

    def similarity(self,trace1,trace2):
        
        raw_trace1=trace1
        raw_trace2=trace2
        
        raw_trace1=raw_trace1[:min(len(trace1),len(trace2))]
        raw_trace2=raw_trace2[:min(len(trace1),len(trace2))]
        
        arr_trace1=np.array(raw_trace1)
        arr_trace2=np.array(raw_trace2)
        
        
        comparison=arr_trace1!=arr_trace2
        equal_array=1*comparison.all()
        return sqrt(np.sum(equal_array)**2+(len(trace1)-len(trace2))**2)
        
    


if __name__=='__main__':
    pass
    # from ucdc import UsageCoverageDrivenClustering

    # datapath='../data/'
    # dataset_names=["scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution"]
    # execution_traces_traceset=load_traceset(datapath,dataset_names[0])


    # clustering_pip=KmeansW2v()
    # sample_heuri=Sampling()

    # u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=execution_traces_traceset)
    # results=u.finetuning_stage(epsilon=0.05,range_clusters=range(2,10),repeat_experiments=2)
    # print(results)

