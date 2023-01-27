# Regression Test Generation by Usage Coverage Driven Clustering on User Traces

This paper is an implementation of Regression Test Generation by Usage Coverage
Driven Clustering on User Traces

## Installation
Install packages with
```bash
pip -r requirements.txt
```
## Run Experiments
```bash
cd ./experiments
python3 ucdc_experiments.py
```

## Benchmark your own Clustering Pipeline and Sampling Strategy
Two abstract classes are available in ./experiments/ucdc/utils. Create your clustering Pipeline and your sampling strategy as classes that implements those abstract class and add them to ucdc_experiments.py
```python
from abc import ABC, abstractmethod

class ClusteringPipeline(ABC):

    

    @abstractmethod
    def preprocessor(self,execution_traces_agilkia_format):
        pass
    
    @abstractmethod
    def fit_predict(self,preprocessed_execution_traces,k):
        pass

class SampleHeuristic(ABC):


    @abstractmethod
    def tests_extraction(self,execution_traces_agilkia_format,cluster_labels):
        pass
```
## Results
Plots and json results are in ./experiments/results.
Such as
### Scanner Results
![Scanner results](https://github.com/frederictamagnan/AIST2023UCDC/blob/main/experiments/results/fig/experiments_scanette_100043-steps_final_2.png)
### Spree Results
![Spree results](https://github.com/frederictamagnan/AIST2023UCDC/blob/main/experiments/results/fig/experiments_spree_5000_session_wo_responses_agilkia_final_2.png)
### Teaming Results
![Teaming results](https://github.com/frederictamagnan/AIST2023UCDC/blob/main/experiments/results/fig/experiments_teaming_execution_final_2.png)
## License

[MIT](https://choosealicense.com/licenses/mit/)