from .utils.clustering_pipeline import ClusteringPipeline
from .utils.sample_heuristic import SampleHeuristic
from .utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
from random import sample

class KmeansPipeline(ClusteringPipeline):
    
    def __init__(self):
        pass
    def preprocessor(self, execution_traces_agilkia_format):
        
        textset=traceset_to_textset(execution_traces_agilkia_format)
        X,voc=textset_to_bowarray(textset) 
        X = StandardScaler().fit_transform(X)
        
        return X

    def fit_predict(self, preprocessed_execution_traces,k):
        model=KMeans(n_clusters=k)
        model.fit(preprocessed_execution_traces)
        return model.fit_predict(preprocessed_execution_traces)

class Sampling(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):

        listset=traceset_to_textset(execution_traces_agilkia_format,start_end_token_creator=lambda i:(list(),list()),format='lst')
        #arrange execution traces index by cluster id in a dict
        idx_by_c=defaultdict(list)
        for idx,c in enumerate(cluster_labels):
            idx_by_c[c].append(idx)
        tests_idx=[]
        #sample one trace index for each cluster
        for c,list_of_idx in idx_by_c.items():
            tests_idx.append(sample(list_of_idx,1)[0])
        testset=[]
        #extract the corresponding traces in a test set
        for idx in tests_idx:
            testset.append(listset[idx])
        return testset

if __name__=='__main__':
    from ucdc import UsageCoverageDrivenClustering

    datapath='../data/'
    dataset_names=["scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution"]
    execution_traces_traceset=load_traceset(datapath,dataset_names[0])


    clustering_pip=KmeansPipeline()
    sample_heuri=Sampling()

    u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=execution_traces_traceset)
    results=u.finetuning_stage(epsilon=0.05,range_clusters=range(2,10),repeat_experiments=2)
    print(results)

