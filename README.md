
# torch-fl
This project gives you an easy to use and extensible setup for Federated Learning and Backdoor Attacks on it.  It has implementations of BlackBox and PGD attacks with and without model replacement. Its modular design allows you to add new models, datasets etc. quickly and  you can easily run several experiments (for tuning etc.) in parallel using just the config files.

The attacks are not tightly coupled with the system so you can use it to do normal federated learning as well. If you want to add any new model, dataset or algorithm then, this system allows you to integrate and test this incrementally by first writing code for single worker setting and then you only need to write appropriate config files and shard the dataset to enable federated learning for your model and datasets. 
	
As an illustration  we provide code for doing federated learning and edge case backdoor attacks on Femnist and Sentiment-140 datasets. This setup was used to obtain results on Sentiment-140 dataset for the following paper,

#### Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
   https://papers.nips.cc/paper/2020/file/b8ffa41d4e492f0fad2f13e29e1762eb-Paper.pdf

### Depdendencies (tentative)
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1
* NLTK 3.4
* preprocessor 1.1.3

### Data Preparation
---
1. For Sentiment140 dataset, please download the dataset from http://help.sentiment140.com/for-students and use the script `./src/datasets/build_twitter_dataset.py` to extract and pre-process a 25% random sample. The backdoor tweets are provided in `./data/sentiment-140/greek-director-backdoor`. Please preprocess and build it using `./src/datasets/build_twitter_backdoor.py`.

### Running Experients:
---
The main script is `./fl_runner.py`, to run various experiments like defenses, edge case vs non-edge case, we provide separate scripts which can run different hyper-parameter settings either sequentially or in parallel depending on the resource availability. Following is detailed description on the configuration parameters which need to be set appropriately for each experiment.
To run `fl_runner.py` please use the following command
`python fl_runner.py --config <config file path>`

### Parameters Description 
Please refer to file `./fl-configs/sent140-fl-conf-greek-director-backdoor.yaml` for detailed description of configuration parameters.

