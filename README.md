# QnA Chatbot with Retrieval-Augmented Generation (RAG) and FAISS

This repository contains the implementation of a sophisticated **multi-turn question-answering (QnA) chatbot**. The chatbot uses **Retrieval-Augmented Generation (RAG)** architecture, combining deep learning models for both passage retrieval and answer generation. The system leverages **FAISS** for efficient document retrieval and **Hugging Face models** for natural language understanding and generation. The project is built using **Streamlit** for the user interface and deployed via **ngrok**.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [References](#references)

## Introduction
This project implements a QnA chatbot that can handle multi-turn interactions by utilizing a combination of **transformer-based models** and **FAISS** for document retrieval. It is built using the **Stanford Question Answering Dataset (SQuAD v2.0)**, which contains both answerable and unanswerable questions. The system's goal is to retrieve relevant information from the dataset and generate accurate and context-aware answers using the **Retrieval-Augmented Generation (RAG)** model.

## Features
- **Multi-turn Conversations**: Keeps track of chat history for smooth QnA flow across multiple turns.
- **Efficient Document Retrieval**: Utilizes **FAISS** for fast retrieval of contextually relevant passages.
- **Answer Generation**: Uses transformer-based models to generate human-like responses.
- **User Interface**: Interactive **Streamlit** interface for users to ask questions and view retrieved documents.
- **Real-time Visualization**: Displays document embeddings and similarity scores for each query.

## System Architecture
The architecture follows a **Retrieval-Augmented Generation (RAG)** approach, which involves two main components:
1. **Retrieval**: 
   - Uses FAISS to index and retrieve relevant document chunks from the dataset.
   - Pre-trained models like `sentence-transformers/all-MiniLM-L6-v2` are used to embed the documents into vectors.
   
2. **Generation**:
   - Retrieved passages are passed to transformer-based models like **BERT**, **RoBERTa**, and **ALBERT** to generate relevant answers.

3. **Interactive UI**:
   - The user interacts with the chatbot using **Streamlit** for input/output.
   - The chatbot also provides visualizations of document embeddings and similarity scores.

## Methodology
1. **Data Preprocessing**: 
   - The SQuAD v2.0 dataset is split into manageable chunks using **LangChain**'s `RecursiveCharacterTextSplitter`.
   
2. **Document Retrieval**: 
   - FAISS is used to index the document chunks by embedding them into vectors, allowing fast similarity searches.
   
3. **Answer Generation**: 
   - The **RAG** model first retrieves relevant context using FAISS, then generates answers based on the retrieved documents.
   
4. **Evaluation Metrics**: 
   - The system is evaluated based on metrics like **Exact Match (EM)**, **F1-Score**, **Retrieval Precision**, **Retrieval Recall**, and **Generation Accuracy**.

## Usage
You can interact with the chatbot by asking it a question via the **Streamlit** interface. The system will retrieve relevant documents and generate answers based on them.

### Example Queries:
```
User: What is the capital of France?
Bot:  Paris

User: What is its population?
Bot:  The population of Paris is 2,241,346 in its administrative city limits as of January 2014. However, the entire Paris agglomeration has a population of 10,550,350 and the Paris metropolitan area has a population of 12,341,418 as of the January 2012 census.

User: What language is spoken there?
Bot:French

User: What is the best time to plan for a holiday visit?
Bot:  The best time to plan for a holiday visit to Paris would be during the Paris Fashion Week in January or July, as well as during the summer months when the weather is pleasant and there are many cultural events and festivals happening in the city.

User: How to get there from San Diego?
Bot:  You can travel from San Diego to Paris by taking a flight from San Diego International Airport to Paris Charles de Gaulle Airport. Alternatively, you can take a train from San Diego to Los Angeles and then transfer to a flight to Paris.

User: exit
Goodbye!
```


## Future Improvements
- **Dynamic Knowledge Base**: Support for real-time, dynamic data sources.
- **Advanced Ranking Mechanism**: Implement more advanced document ranking models for improved retrieval.
- **Scalability**: Improve the system to handle larger datasets and complex multi-turn conversations.

## References
- Murphy, K. P. (2023). *Probabilistic Machine Learning: An Introduction*. The MIT Press.
- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Association for Computational Linguistics.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Association for Computational Linguistics.
