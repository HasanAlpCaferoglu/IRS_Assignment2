# CS533 Information Retrieval System Assignment 2

### Course No: CS533
### Assignment No: 2
### Student Name: Hasan Alp Caferoğlu
### Student Id: 22203991
### Email Address: alp.caferoglu@bilkent.edu.tr

## Overview

* This cs533_assignment2.ipynb file contains the implementation of an Information Retrieval System using a combination of TF-IDF, BM25, and Word Embeddings. The system is designed to retrieve relevant documents for a given query from the CISI dataset. 

* For a determined list of set of weight, the cs533_assignment2.ipynb file computes MAP values for each system. Note that for a set of weights (BM25_score_weight, TF-IDF_score_weight, embedding_score_weigh), the written code calculates MAP for both a system using pre-trained Word2Vec model "word2vec-google-news-300" and model trained by CISI dataset.

## Setup

1. **Install Required Packages:**
   Ensure that the necessary Python packages are installed by running the following commands:

   ```bash
   !pip install rank-bm25
   !pip install ipython-autotime
   %load_ext autotime
