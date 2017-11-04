# DeepTextsummarizer
Deep-Learning-project 2017


Intent is to use deep neural network for abstractive text summarization.
There are two methods in neural based text summarization: 1) Extraction based 2) Abstraction based.
1) Extraction: Important or relevant sentences are learned from the given article or text by the neural network. 
  Those sentences are then combined to form a summary at the end.
2) Abstractive: It is more human like. How a human reads a certain article and how he would write its summary.
  Its more complex and difficult compared to 1. We hope to use local attention based model and sequence2sequence model.
  
# Dataset
We are using a data-driven neural net approach to fulfill text abstractive summarization. 
1) http://duc.nist.gov/data.html is very popular dataset among the NLP community. 500 news articles
2) Amazon food review dataset is available. A lot of kaggle competetions have also used that data set. Size: 251 mb
3)Another dataset in the gigaword datset ,which contains 10 million article but it is not avaliable in the open source community and very difficult to acquire.

Our first choice is 1 which gives access to DUC dataset after completing some organization and individual agreements
If we are not able to get it within time we will use 2.

# Methods to be used
-> Till now we have concluded that we will be using a Recurent Nueral Network(RNN) with Long Short Term Memory(LSTM) encoder and decoder and build a sequence to sequence model. 

# Environment Setup (Machine and Cloud)
-> All of our work will be conducted on google cloud platform.
-> Account has been setup and a VM instance is created on the compute engine.
-> VM instance is setup with 8 cores ,24 GB memory and Nvidia Tesla K80 GPU.
-> All necessary software have been installed ,mainly including python,gpu version tensorflow and all its dependencies.
-> Connection has been setup with the local machine using google cloud compute tools to build a ssh connection and use the VM.
-> Jupyter notebook has been linked from the host PC to the remote VM to allow ease of acess.
-> A cloud storage bucket has been created to store the necessary file on google cloud to use from the VM and provide a redundancy.


