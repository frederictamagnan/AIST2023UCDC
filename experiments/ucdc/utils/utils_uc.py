from .usage_coverage import UsageCoverage
from .utils_preprocessing import get_a_session

def get_usage_coverage_per_cluster(num_clusters,traceset_global,execution_clusters,test_clusters):
    traces_execution_test_per_cluster={}
    usage_per_cluster={}
    for i in range(num_clusters):
        traces_execution_test_per_cluster[i]={"execution":[],"test":[]}
        for idx,label in enumerate(execution_clusters+test_clusters):
            if label==i:
                if idx<len(execution_clusters):
                    traces_execution_test_per_cluster[i]["execution"].append(get_a_session(idx,traceset_global))
                else:
                    traces_execution_test_per_cluster[i]["test"].append(get_a_session(idx, traceset_global))

        if len(traces_execution_test_per_cluster[i]["test"])==0:
            uc = UsageCoverage(traces_execution_test_per_cluster[i]["execution"], list_n_grams=[1, 2, 3,4])
            usage_per_cluster[i]={}
            usage_per_cluster[i]['usage']=0
            usage_per_cluster[i]['missing']=[]
            usage_per_cluster[i]['present']=[]
        elif len(traces_execution_test_per_cluster[i]["execution"])==0:
            uc = UsageCoverage(traces_execution_test_per_cluster[i]["execution"], list_n_grams=[1, 2, 3,4])
            usage_per_cluster[i]={}
            usage_per_cluster[i]['usage']=1
            usage_per_cluster[i]['missing']=[]
            usage_per_cluster[i]['present']=[]

        else:
            uc = UsageCoverage(traces_execution_test_per_cluster[i]["execution"], list_n_grams=[1, 2, 3,4])
            u=uc.usage(traces_execution_test_per_cluster[i]["test"])
            usage_per_cluster[i]={}
            usage_per_cluster[i]['usage']=round(u[0], 2)
            usage_per_cluster[i]['missing']=u[1]
            usage_per_cluster[i]['present']=u[2]


    
    traces_execution_test_per_cluster[-1]={"execution":[],"test":[]}
    for idx,label in enumerate(execution_clusters+test_clusters):
        
        if idx<len(execution_clusters):
            traces_execution_test_per_cluster[-1]["execution"].append(get_a_session(idx,traceset_global))
        else:
            traces_execution_test_per_cluster[-1]["test"].append(get_a_session(idx, traceset_global))

    if len(traces_execution_test_per_cluster[-1]["test"])==0:
        uc = UsageCoverage(traces_execution_test_per_cluster[-1]["execution"], list_n_grams=[1, 2, 3,4])

        usage_per_cluster[-1]={}
        usage_per_cluster[-1]['usage']=0
        usage_per_cluster[-1]['missing']=[]
        usage_per_cluster[-1]['present']=[]
    elif len(traces_execution_test_per_cluster[-1]["execution"])==0:
        uc = UsageCoverage(traces_execution_test_per_cluster[-1]["execution"], list_n_grams=[1, 2, 3,4])

        usage_per_cluster[-1]={}
        usage_per_cluster[-1]['usage']=1
        usage_per_cluster[-1]['missing']=[]
        usage_per_cluster[-1]['present']=[]

    else:
        uc = UsageCoverage(traces_execution_test_per_cluster[-1]["execution"], list_n_grams=[1, 2, 3,4])
        # usage_per_cluster[-1]=round(uc.usage(traces_execution_test_per_cluster[-1]["test"])[0], 2)
        u=uc.usage(traces_execution_test_per_cluster[-1]["test"])
        usage_per_cluster[-1]={}
        usage_per_cluster[-1]['usage']=round(u[0], 2)
        usage_per_cluster[-1]['missing']=u[1]
        usage_per_cluster[-1]['present']=u[2]
    
    return usage_per_cluster