
from ucdc.ucdc import UsageCoverageDrivenClustering
from ucdc.utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray

from ucdc.kmeans_with_sampling import Sampling
from ucdc.agglutinate import AgglutinatePipeline
from ucdc.kmedoids_pipeline import KMedoidsPipeline
from ucdc.kmeans_with_w2v import KmeansW2v
from ucdc.no_clustering_only_sampling import NoClustering,OnlySampling
from ucdc.kmeans_with_best_usage_choice import KmeansPipeline,BestUsageChoice
from ucdc.kmeans_with_sampling import Sampling
import json

from utils_experiments import load_config_dict

import matplotlib.pyplot as plt
from math import sqrt
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# import warnings
# warnings.filterwarnings("ignore")

class UCDCExperiment:
    def __init__(self,name_experiment,result_path="./results/json/results",):
        self.config_dict=load_config_dict()
        self.results={}
        self.result_path=result_path+name_experiment+'.json'

    
    def experiment(self,clustering_pip,sample_heuri):
        for dataset_name in self.config_dict.keys():
            datapath=self.config_dict[dataset_name]['datapath']
            rcd=self.config_dict[dataset_name]["range"]
            experiments=self.config_dict[dataset_name]["experiments"]

            traceset=load_traceset(datapath,dataset_name)
    
            u=UsageCoverageDrivenClustering(clustering_pip,sample_heuri,execution_traces_agilkia_format=traceset)
            self.results[dataset_name]=u.finetuning_stage(epsilon=0.05,range_clusters=range(*rcd),repeat_experiments=experiments)
            with open(self.result_path, "w") as outfile:
                json.dump(self.results, outfile)
    
    def plot_multiple(self,results):
        
        
        datasets=list(results[list(results.keys())[0]].keys())
        colors=['b','r','k','g','c','m','y']
        for dataset in datasets:
            plt.figure()
           
            color_index=0
            for experiment,result in results.items():
                print(result)
                c=colors[color_index]
                plot_results=result[dataset]['usage_coverage_by_clusters']
                x=[a for a,b,c in plot_results]
                y=[b for a,b,c in plot_results]
                z=[c for a,b,c in plot_results]
                w=[1.96*c/sqrt(30) for a,b,c in plot_results]

                sns.set()

                
                mean_1 = np.array(y)
                std_1 = np.array(w)
                # plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
                plt.plot(x, mean_1, 'b-', label=experiment,color=c)
                plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color=c, alpha=0.2)
                
                color_index+=1
                # plt.legend(title='Usage Coverage per # of clusters/tests',prop={'size': 8})
                plt.legend(prop={'size': 8.25})

                plt.show()

            # plt.savefig("./results/fig/experiments_"+dataset+"_final_2.png",format="png")
            plt.savefig("./results/fig/experiments_"+dataset+"_final_2.svg",format="svg")


if __name__=='__main__':
    
    experiment=False
    plot=True
    
    if experiment:
        results_path=[]
        pipelines=[("agglutinate",AgglutinatePipeline,Sampling)]
        pipelines=[("NoClusteringOnlySampling",NoClustering,OnlySampling),("KmeansWithSampling",KmeansPipeline,Sampling),("KmeansWithBestUsageChoice",KmeansPipeline,BestUsageChoice)]
        pipelines+=[("kmedoids",KMedoidsPipeline,Sampling),("kmeansW2v",KmeansW2v,Sampling),('agglutinate',AgglutinatePipeline,Sampling)]
        pipelines=[("kmeansW2vBestUsageChoice",KmeansW2v,BestUsageChoice)]
        for pipeline in pipelines:
            name_experiment,clustering_pip,sample_heuri=pipeline
            ucdce=UCDCExperiment(name_experiment+"2001")
            ucdce.experiment(clustering_pip(),sample_heuri())
            results_path.append((name_experiment,ucdce.result_path))
    
    if plot:
        
        results={}
        
        for result_path in results_path:
            with open(result_path[1]) as json_file:
                results[result_path[0]]= dict(json.load(json_file))
                
        ucdce=UCDCExperiment("plot")
        ucdce.plot_multiple(results)
    
    
