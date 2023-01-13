from .utils.clustering_pipeline import ClusteringPipeline
from .utils.sample_heuristic import SampleHeuristic
from .utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray,get_a_session
from .utils.usage_coverage import UsageCoverage


from copy import deepcopy
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

class BestUsageChoice(SampleHeuristic):
    def __init__(self):
        pass
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):
        num_clusters=max(cluster_labels)+1
        traces_execution_per_cluster={}
        n_grams_per_cluster={}
        tests=[]
        tests_idx=[]
        for i in range(num_clusters):
            traces_execution_per_cluster[i]=[]
            for idx,label in enumerate(cluster_labels):
                if label==i:
                
                    traces_execution_per_cluster[i].append(get_a_session(idx,execution_traces_agilkia_format))
                    
            
        
            uc = UsageCoverage(traces_execution_per_cluster[i], list_n_grams=[1, 2, 3,4])
            
            list_n_grams=[(n_gram,freq) for n_gram,freq in uc.reference_n_grams_weighted.items()]
            n_grams_per_cluster[i]=sorted(list_n_grams, key=lambda tup: tup[1],reverse=True)
            candidates=[]
            
            candidates=[]
            for trace in traces_execution_per_cluster[i]:
                candidates.append(','.join(trace))
            
            frozen_candidates=[None]
            frozen_candidates_idx=[]
            candidates_idx=list(range(len(candidates)))
            idx_n_gram=0
            while len(candidates)>0 and idx_n_gram<len(n_grams_per_cluster[i]):
                
                wrong_indexes=[]
                frozen_candidates=deepcopy(candidates)
                frozen_candidates_idx=deepcopy(candidates_idx)
                for idx,trace in enumerate(candidates):
                    if n_grams_per_cluster[i][idx_n_gram][0] not in trace:
                        wrong_indexes.append(idx)
                for index in sorted(wrong_indexes, reverse=True):
                    del candidates[index]
                    del candidates_idx[index]
                
                idx_n_gram+=1
            
            print("found "+str(len(frozen_candidates))+" candidates for cluster "+str(i)+" and reached the "+str(idx_n_gram)+"-th n gram")
            tests+=[test.split(',') for test in frozen_candidates[:1]]

           
            tests_idx=frozen_candidates_idx
        return tests

if __name__=='__main__':
    from ucdc import UsageCoverageDrivenClustering

    datapath='../data/'
    dataset_names=["scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution"]
    execution_traces_traceset=load_traceset(datapath,dataset_names[0])


    clustering_pip=KmeansPipeline()
    sample_heuri=BestUsageChoice()

    u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=execution_traces_traceset)
    results=u.finetuning_stage(epsilon=0.05,range_clusters=range(2,30),repeat_experiments=2)
    print(results)

