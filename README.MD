# Instructions

1) Set the neuronpedia API KEY in experiment.py
2) Define your experiments in experiment.py start method.
3) step1.py will generate and download the attribution graphs.
4) step2.py will find the relevant neuron features
5) step3.py will steer the model responses.

The attribution graphs, request cache and results will be written in the data folder. By default these are fully pushed to git to enhance reproducibility.

To generate the results report you can run the extra_analysis.py (after step2 is complete) and  extra_steering.py (after step3 is complete).