from statistics import mean,stdev
from .utils.usage_coverage import UsageCoverage
from tqdm import tqdm
from .logging_manager import LoggingManager
class UsageCoverageDrivenClustering:
    
    def __init__(self,clustering_pipeline,sample_heuristic,execution_traces_agilkia_format):
        
        
        self.clustering_pipeline = clustering_pipeline
        self.sample_heuristic=sample_heuristic
        self.execution_traces_agilkia_format=execution_traces_agilkia_format
        
        self.uc=UsageCoverage(self.execution_traces_agilkia_format,list_n_grams=[1,2,3,4])
        
        

    def finetuning_stage(self,range_clusters=range(2,20),epsilon=0.05,repeat_experiments=20):
        usage_coverage_stats=0
        recorded_usage_coverage_stats=[]
        results={}
        
        X=self.clustering_pipeline.preprocessor(self.execution_traces_agilkia_format)
        
        for nb_clusters in range_clusters:
            print("nb cluster "+ str(nb_clusters))
            usage_experiments=[]
            for experiment in range(repeat_experiments):
                print("--------nb experiment :  "+ str(experiment))
                tests=self.clustering_and_heuristic(X,nb_clusters)
                usage_experiments.append(self.compute_usage(tests))
            
            usage_coverage_stats=(mean(usage_experiments),stdev(usage_experiments))
            print("--usage coverage: "+ str(usage_coverage_stats))
            recorded_usage_coverage_stats.append(usage_coverage_stats)
            if 1-usage_coverage_stats[0]<epsilon:
                epsilon=-100
                results['best_nb_of_clusters']=nb_clusters
        
        results['usage_coverage_by_clusters']=[(a,b,c) for (a,(b,c)) in zip(list(range_clusters),recorded_usage_coverage_stats)]
        return  results
                
    
    def clustering_and_heuristic(self,X,nb_clusters):
        cluster_labels=self.clustering_pipeline.fit_predict(X,nb_clusters)
        tests=self.sample_heuristic.tests_extraction(self.execution_traces_agilkia_format,cluster_labels)
        return tests 
    
    def compute_usage(self,tests):
        usage,missing_n_grams,present_n_grams=self.uc.usage(tests)
        return usage