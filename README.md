# Deep-Project_Textsummarizer
Deep-project 2017


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
3) Another world famous dataset is gigaword dataset. It costs about 3000$ so it is out of the picture. 10 million news articles

Our first choice is 1 which gives access to DUC dataset after completing some organization and individual agreements
If we are not able to get it within time we will use 2.

# Computation Engine
All of our work will be conducted on google cloud platform. A student account can be made free of cost.
It gives a free credit of 300$. It meets all the computational power we require.
