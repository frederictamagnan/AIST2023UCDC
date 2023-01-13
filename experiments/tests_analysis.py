from sklearn.preprocessing import StandardScaler
from ucdc.utils.utils_preprocessing import load_traceset,traceset_to_textset,textset_to_bowarray,get_a_session
from sklearn.cluster import KMeans
from ucdc.utils.usage_coverage import UsageCoverage
import matplotlib.pyplot as plt
import copy
from collections import Counter
import numpy as np
from ucdc.utils.utils_uc import get_usage_coverage_per_cluster
import seaborn as sns
#config
datapath='../data/'
# dataset_names=["1026-steps","scanette_100043-steps","spree_5000_session_wo_responses_agilkia","teaming_execution"]
dataset_names=["scanette_100043-steps","teaming_execution"]
# dataset_names=["teaming_execution"]
# dataset_names=["teaming_execution"]
test_dataset_names=['functional-tests',"teaming_test"]
# test_dataset_names=["teaming_test"]

cluster_number=[20,90]

for dataset_id in range(0,2):
    dataset_name=dataset_names[dataset_id]
    test_dataset_name=test_dataset_names[dataset_id]

    #load dataset
    traceset_execution=load_traceset(datapath,dataset_name)
    textset_execution=traceset_to_textset(traceset_execution)
    listset_execution=traceset_to_textset(traceset_execution,start_end_token_creator=lambda i:(list(),list()),format='lst')
    X_execution,voc_execution=textset_to_bowarray(textset_execution)
    # print(len(voc_execution))
    traceset_test=load_traceset(datapath,test_dataset_name)
    textset_test=traceset_to_textset(traceset_test)
    listset_test=traceset_to_textset(traceset_test,start_end_token_creator=lambda i:(list(),list()),format='lst')
    # X_test,voc_test=textset_to_bowarray(textset_test,vocabulary_provided=voc_execution)
    X_test,voc_test=textset_to_bowarray(textset_test,vocabulary_provided=None)

    # print(len(voc_test))
    new_data = np.unique(X_test, axis=0)
    # print("unique",len(new_data))
    traceset_global=copy.deepcopy(traceset_execution)
    traceset_global.extend(traceset_test.traces)
    textset_global=traceset_to_textset(traceset_global)
    listset_global=traceset_to_textset(traceset_global,start_end_token_creator=lambda i:(list(),list()),format='lst')
    X_global,voc_global=textset_to_bowarray(textset_global,vocabulary_provided=voc_execution)


    scaler = StandardScaler().fit(X_execution)
    X_execution=scaler.transform(X_execution)
    X_global=scaler.transform(X_global)
    num_clusters=cluster_number[dataset_id]
    kmeanModel = KMeans(n_clusters=num_clusters).fit(X_execution)
    y=kmeanModel.predict(X_global)
    execution_clusters=y[:len(traceset_execution)].tolist()
    test_clusters=y[len(traceset_execution):].tolist()
    
    data=get_usage_coverage_per_cluster(num_clusters,traceset_global,execution_clusters,test_clusters)
    # print(data.keys())
    # print(data[-1].keys())
    counts = Counter(y)
    count_pairs = sorted(list(counts.items()))
    # print(count_pairs)
    execution_len=[]
    for i in range(num_clusters):
        execution_len.append(count_pairs[i][1])
    

    usage_len=[]
    for i in range(num_clusters):
        usage_len.append(data[i]['usage']*100)

    labels = list(range(num_clusters))
   
    
    print("GLOBAL",data[-1]['usage'])
    
    for cluster,usage_info in data.items():
        if usage_info['usage']<0.9 and usage_info['usage']>0.1:
            print(cluster,usage_info['usage'])
            data_missing_piechart=[]
            labels_missing_piechart=[]
            data_present_piechart=[]
            labels_present_piechart=[]
            for ngram,usage_gram in usage_info['missing']:
                if usage_gram>0.1:
                    data_missing_piechart.append(usage_gram)
                    labels_missing_piechart.append(ngram)
            for ngram,usage_gram in usage_info['present']:
                if usage_gram>0.1:
                    data_present_piechart.append(usage_gram)
                    labels_present_piechart.append(ngram)
            
            # print(data_missing_piechart,labels_missing_piechart,data_present_piechart,labels_present_piechart)
            
            if len(data_missing_piechart)>0 and len(data_present_piechart)>0:
            #define Seaborn color palette to use
                plt.figure()
                colors = list(reversed(sns.color_palette('Reds')[0:len(data_missing_piechart)]))+list(reversed(sns.color_palette("Greens")[0:len(data_present_piechart)]))
                print(len(colors))
                data_piechart=data_missing_piechart+data_present_piechart
                data_piechart=[str(elt) for elt in data_piechart]
                labels_piechart=labels_missing_piechart+labels_present_piechart
                labels_piechart=[elt[0:15]+'...\n...'+elt[-15:] for elt in labels_piechart]
                print(len(data_piechart))
                print(data_piechart,labels_piechart)
                #create pie chart
                plt.pie(data_piechart, labels=labels_piechart, colors = colors, autopct='%.0f%%')
                if cluster!=-1:
                    plt.title("Api Calls Usage Share For cluster "+str(cluster))
                else:
                    plt.title("Global Api Calls Usage Share "+str(cluster))

                plt.show()
                plt.savefig("./results/fig/piecharts/"+dataset_name+"_cluster_"+str(cluster)+"with_usage_"+str(usage_info['usage'])+".png")
            
            
            
            
            # print(cluster,usage_info['usage'])
            # print(usage_info['missing'][:5])
            # print(usage_info['present'][:5])
    
    font_color = '#525252'
    hfont = {'fontname':'Calibri'}
    facecolor = '#eaeaf2'
    color_red = '#fd625e'
    color_blue = '#01b8aa'
    title0="Number of sessions (#)"
    title1="Usage Coverage (%)"
    index=labels
    font_color = '#525252'


    fig, axes = plt.subplots(figsize=(10,20), facecolor=facecolor, ncols=2,sharey=True)
    
    fig.tight_layout()

    axes[0].barh(index, execution_len, align='center', color=color_red, zorder=10,log=True)
    axes[0].set_title(title0, fontsize=18, pad=15, color=color_red, **hfont)
    axes[1].barh(index, usage_len, align='center', color=color_blue, zorder=10)
    axes[1].set_title(title1, fontsize=18, pad=15, color=color_blue, **hfont)
   

    # If you have positive numbers and want to invert the x-axis of the left plot
    axes[0].invert_xaxis() 


    # To show data from highest to lowest
    plt.gca().invert_yaxis()

    axes[0].set(yticks=labels, yticklabels=labels)
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', colors='white') # tick color

    axes[1].set_xticks([0,20,40,60,80,100])
    axes[1].set_xticklabels(['0','20', '40', '60', '80', '100'])


    for label in (axes[0].get_xticklabels() + axes[0].get_yticklabels()):
        label.set(fontsize=13, color=font_color, **hfont)
    for label in (axes[1].get_xticklabels() + axes[1].get_yticklabels()):
        label.set(fontsize=13, color=font_color, **hfont)
        
    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.show()

    plt.savefig("./results/fig/barplot"+dataset_name+"_usage_coverage.jpg")



    