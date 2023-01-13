from .utils_ngrams import n_gram_coverage,extract_all_n_grams,extract_all_n_grams_weighted
from statistics import mean


class UsageCoverage:
    def __init__(self,reference_traceset,dataset_type='scanette',list_n_grams=[1,2,3,4],nb_of_traces_generated=[10, 30, 50],nb_of_experiments=100):
        self.reference_traceset=reference_traceset
        self.dataset_type=dataset_type
        self.list_n_grams=list_n_grams
        self.nb_of_experiments=nb_of_experiments
        self.reference_n_grams_len=[]
        self.nb_of_traces_generated=nb_of_traces_generated
        for n in self.list_n_grams:
            self.reference_n_grams_len.append(len(extract_all_n_grams(self.reference_traceset, n=n)))
        self.reference_n_grams_weighted={}
        for n in self.list_n_grams:
            self.reference_n_grams_weighted.update(extract_all_n_grams_weighted(self.reference_traceset, n=n))



    def n_gram_coverage(self,candidate_traceset):
        candidate_traceset_n_grams_len=[]
        for n in self.list_n_grams:
            candidate_traceset_n_grams_len.append(len(extract_all_n_grams(candidate_traceset, n=n)))
        return sum(candidate_traceset_n_grams_len)/sum(self.reference_n_grams_len)

    def missing_grams(self, candidate_n_gram_weighted, total_usage):
        list_n_grams = []
        for n_gram in self.reference_n_grams_weighted:
            if n_gram not in candidate_n_gram_weighted:
                list_n_grams.append((n_gram, self.reference_n_grams_weighted[n_gram] / total_usage))

        return sorted(list_n_grams, key=lambda tup: tup[1],reverse=True)
    def present_grams(self, candidate_n_gram_weighted, total_usage):
        list_n_grams = []
        for n_gram in self.reference_n_grams_weighted:
            if n_gram in candidate_n_gram_weighted:
                list_n_grams.append((n_gram, self.reference_n_grams_weighted[n_gram] / total_usage))

        return sorted(list_n_grams, key=lambda tup: tup[1],reverse=True)

    def usage(self,candidate_traceset):
        candidate_n_grams_weighted = {}
        for n in self.list_n_grams:
            candidate_n_grams_weighted.update(extract_all_n_grams_weighted(candidate_traceset, n=n))

        usage=0
        total_usage=sum([v for k,v in self.reference_n_grams_weighted.items()])
        for n_grams in candidate_n_grams_weighted.keys():
            if n_grams in self.reference_n_grams_weighted.keys():
                usage+=self.reference_n_grams_weighted[n_grams]
        
        
        return usage/total_usage,self.missing_grams(candidate_n_grams_weighted,total_usage),self.present_grams(candidate_n_grams_weighted,total_usage)






