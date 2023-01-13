from .utils.clustering_pipeline import ClusteringPipeline
from .utils.sample_heuristic import SampleHeuristic
from .utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample
from pprint import pprint

class NoClustering(ClusteringPipeline):
    
    def __init__(self):
        pass
    def preprocessor(self, execution_traces_agilkia_format):
        return execution_traces_agilkia_format

    def fit_predict(self, preprocessed_execution_traces,k):
        return [k for i in range(1)]

class OnlySampling(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):
        nb_of_tests=cluster_labels[0]
    
        cluster_labels=[0 for i in range(len(execution_traces_agilkia_format))]
        listset=traceset_to_textset(execution_traces_agilkia_format,start_end_token_creator=lambda i:(list(),list()),format='lst')
        tests_idx=sample(list(range(len(execution_traces_agilkia_format))),nb_of_tests)
        testset=[]
        for idx in tests_idx:
            testset.append(listset[idx])

        return testset

# if __name__=='__main__':
#     from ucdc import UsageCoverageDrivenClustering

#     datapath='../data/'
#     dataset_names=["scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution"]
#     execution_traces_traceset=load_traceset(datapath,dataset_names[0])


#     clustering_pip=NoClustering()
#     sample_heuri=Sampling()

#     u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=execution_traces_traceset)
#     results=u.finetuning_stage(epsilon=0.05,range_clusters=range(2,30),repeat_experiments=3)
#     print(results)

