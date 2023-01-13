
from ucdc.utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray,get_a_session
from utils_experiments import load_config_dict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statistics import mean

import json


class TraditionalMetricExperiment:
    
    def __init__(self,name_experiment,metric,result_path="./results/json/results",):
        self.config_dict=load_config_dict()
        self.results={}
        self.result_path=result_path+name_experiment+'.json'
        self.metric=metric
    
    
    def experiment(self):
        for dataset_name in self.config_dict.keys():
            datapath=self.config_dict[dataset_name]['datapath']
            rcd=self.config_dict[dataset_name]["range"]
            experiments=self.config_dict[dataset_name]["experiments"]

            traceset=load_traceset(datapath,dataset_name)
            textset=traceset_to_textset(traceset)
            X,voc=textset_to_bowarray(textset)
            dbs_means=[]
            
            for c in tqdm(range(*rcd)):
                dbs_l=[]
                for i in range(experiments):
                    print("nb cluster :"+str(c)+" iteration #" + str(i))
        
                    scaler = StandardScaler().fit(X)
                    X=scaler.transform(X)
                    
                    kmeanModel = KMeans(n_clusters=c).fit(X)

                    y=kmeanModel.fit_predict(X)
                    dbs=self.metric(X,y)
                    dbs_l.append(dbs)
                
                dbs_means.append((c,mean(dbs_l)))
            
            self.results[dataset_name]=dbs_means
            
            with open(self.result_path, "w") as outfile:
                json.dump(self.results, outfile)
    
    def plot(self):
        pass
if __name__=='__main__':
    from sklearn.metrics import silhouette_score
    from sklearn.metrics import davies_bouldin_score

    dbe=TraditionalMetricExperiment("silhouette",silhouette_score)
    dbe.experiment()
    

    