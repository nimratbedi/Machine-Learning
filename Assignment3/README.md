# TweetsClusteringUsing_KMeans

Clustering tweets by utlizing Jaccard Distance metric and K-means clustering algorithm.


<hr>

## Approach

- Compute the similarity between tweets using Jaccard Distance Metric
- Cluster tweets using the K-means clustering algorithm

## Libraries used:
numpy and pandas

## Input to K-means algorithm

1. The number of clusters K (Taken K=5,10,15,20,25)
2. A real world dataset sampled from Health news in Twitter that contains more than 3000 tweets. The tweet dataset is in text format and can be found throught this link https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter

## Compile and Run Instructions:

**Commands to run:**

```
1.open python editor
2.Make dataset set directory as your current working directory 
	eg. "/Users/nimratbedi/Downloads/Health-Tweets"
3.You can flexibly change to any dataset and K value.
4.Run the program to find the number of clusters and the SSE value.(python3 Tweet_Clustering.py )


**Results:**
Result can be seen in tablular format in the report.pdf 
