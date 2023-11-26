# CS533 Information Retrieval System Assignment 2

#### Course No: CS533
#### Assignment No: 2
#### Student Name: Hasan Alp Caferoğlu
#### Student Id: 22203991
#### Email Address: alp.caferoglu@bilkent.edu.tr

## Overview

* This cs533_assignment2.ipynb file contains the implementation of an Information Retrieval System using a combination of TF-IDF, BM25, and Word Embeddings. The system is designed to retrieve relevant documents for a given query from the CISI dataset. 

* For a determined list of set of weight, the cs533_assignment2.ipynb file computes MAP values for each system. Note that for a set of weights (BM25_score_weight, TF-IDF_score_weight, embedding_score_weigh), the written code calculates MAP for both a system using pre-trained Word2Vec model "word2vec-google-news-300" and model trained by CISI dataset.

* After program execution, MAP value for each system along with its information, i.e. weights and which word embedding method used, is written to a file.

## Important Notes

* CISI dataset should be uploaded to Google Drive, if the code is directly used without changing anything.
* Google Drive paths should be updated according to the user Google Drive directory structure

## Setup

1. **Install Required Packages:**
   Ensure that the necessary Python packages are installed by running the following commands:

   ```bash
   !pip install rank-bm25
   !pip install ipython-autotime
   %load_ext autotime
   
2. Upload CISI dataset into your Google Drive
3. **Drive Connection:**
   Ensure that you give a permission to Google Colab to access your Drive.
   Update CISI dataset paths according to your director structure 

   ```bash
   from google.colab import drive
   drive.mount('/content/drive')
   CISI_documents_path = '/content/drive/MyDrive/.../CISI/documents.csv'
   CISI_queries_path = '/content/drive/MyDrive/.../CISI/queries.csv'
   CISI_ground_truth_path = '/content/drive/MyDrive/.../CISI/ground_truth.csv'

4. **Imports Required Libraries**
   Make sure that you import all the necessary libraries

   ```bash
   import pandas as pd
   from gensim.models import Word2Vec
   import gensim.downloader as api
   import numpy as np
   from sklearn.feature_extraction.text import TfidfVectorizer
   from rank_bm25 import BM25Okapi
   import nltk
   from nltk.corpus import stopwords
   import string

   # Downloading NLTK resources
   nltk.download('stopwords')

   
5. **Last Modification For Calculating MAP for Various Systems**
   At the last step, ensure that path that results printed in a file is updated according to your Drive directory structure.
   Enter set of weights for BM25 score, TF-IDF score and embedding score respectively.
   Note that system will automatically measure MAP for a configuration both using pre-trained word2vec model and model trained on CISI
   
   ```bash
      # Setting path for printed result.txt file
      systemPerformanceFilePath = '/content/drive/MyDrive/1.MS/CS533_IRS/assignments/assignment2/implementation/resutls.txt'
      # Setting weights to create different configurations
      # Note that system will automatically measure MAP for a configuration both using pre-trained word2vec model and model trained on CISI
      systemsWeightList = [(0.4, 0.25, 0.35), (0.4, 0.25, 0.35)] # List of set of weights

   
```bash
   # Constant embedding model list
   embeddingModels = [model.wv, preTrainedGoogleNewsModel]
   
   systemsDict = {}
   sysCount = 1
   for index in range(len(systemsWeightList)):
     system = {'embeddingModel': 0, 'weights': systemsWeightList[index]}
     systemsDict[f'{sysCount}'] = system
     sysCount+=1

     # if embeddingWeight is larger than 0, there must 2 systems with both our trained model and pretrained model
     if systemsWeightList[index][2] > 0:
       systemWithAnotherModel = {'embeddingModel': 1, 'weights': systemsWeightList[index]}
       systemsDict[f'{sysCount}'] = systemWithAnotherModel
       sysCount+=1
   
   
   # print(systemsDict)
   # print(len(systemsDict))
   for systemId in range(1, len(systemsDict)+1):
     system = systemsDict[f'{systemId}']
     weights = system['weights']
     tfidfWeight, bm25Weight, embeddingWeight = system['weights']
     embeddingModelNo = system['embeddingModel']
     MAP = calculateMAPForSystem(ground_truth_df, documents_df, queries_df, embeddingModels[embeddingModelNo], tfidfWeight, bm25Weight, embeddingWeight)
     system['MAP'] = MAP
     print(f'System {systemId}: \n MAP: {MAP} Weights: {weights} Embedding Model: {embeddingModels[embeddingModelNo]} \n')
   
   
   # Printing System Performances into txt file
   with open(systemPerformanceFilePath, 'w') as fileHandle:
     for systemId, system in systemsDict.items():
       weights = system['weights']
       MAP = system['MAP']
       embeddingModelId = system['embeddingModel']
       if embeddingModelId == 1:
         embeddingModel = 'Trained Model with CISI Dataset'
       else:
         embeddingModel = 'word2vec-google-news-300'
       line = f'System {systemId}: \n MAP: {MAP} Weights: {weights} Embedding Model: {embeddingModel} \n'
       fileHandle.write(line)
   
   
