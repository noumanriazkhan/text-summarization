# Automated Text Summarization

This automated research assistant outputs summarized bullet points of sub-topics by taking keywords input from user and output only related summary as bullet points.

Automated Text summarization tasks can be classified into extractive and abstractive summarization which could be further broken down into Single Document and Multi Document text summarization. Our goal is to generate multi-document abstractive summaries of sub-topics in this Automated Research Assistant (ARA); which is “understanding” the original text and “retelling” it in fewer words. However due to limited timeline I remained stick to the extractive summaries of sub-topics.

## Design Overview

![alt text](https://github.com/noumanriazkhan/text-summarization/blob/master/design.PNG)

## Goals

The design of our ARA could have following goals:

	Reading and preprocessing documents from plain text files which includes tokenization, stop words removal, case change and stemming.

	Document Clustering of input documents to group similar documents in clusters. 

	Topic Modelling due to no label or keyword information, unsupervised technique to be used for topic modelling.

	Topic Input from the user for topics and subtopics.

	Relevant Documents retrieval against input topics and subtopics. The similarity is to be measured between input topic and topic modelling output to identify most relevant cluster.

	Summarization using ‘TextRank’ approach to model text as graph networks and retrieve high importance sentences as summaries.

## File Descripion

The submitted task contains 04 files, details and instructions as following:

**summarizer.py** is the main file which can be run from Command Prompt as 

python summarizer.py

without arguments. On execution it first asks for the path where articles are location, e.g. to build on given articles, ‘./articles/’ is required. Next it asks for Topic and Sub Topics.

**preprocess.py** contains the pre-processing function.

**extractive.py** applies ‘TextRank’ algorithm and contains ‘summarize’ function for extractive summarization.

**summary.py** contains the function for finding similar sentences to relevant to subtopics.
