from .utils.clustering_pipeline import ClusteringPipeline
from .utils.sample_heuristic import SampleHeuristic
from .utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample
import gensim
import numpy as np

class KmeansW2v(ClusteringPipeline):
    
    def __init__(self):
        pass
    def preprocessor(self, execution_traces_agilkia_format):
        listset=traceset_to_textset(execution_traces_agilkia_format,start_end_token_creator=lambda i:(['<sos>'],['<eos>']),format='lst')
        print(listset[0])
        model = gensim.models.Word2Vec(sentences=listset,vector_size=10,window=5,min_count=1)
        means=[]
        for seq in listset:
            vecs=[model.wv[elt] for elt in seq]
            means.append(np.mean(vecs,axis=0))
        X=np.array(means)
        
        return X

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        model.fit(preprocessed_execution_traces)
        return model.fit_predict(preprocessed_execution_traces)



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

