## Data
The data can be taken from the ecommerce app we already have. If there's no data yet, we can crawl our data in another web or get it legally in Open Data platform like Kaggle.

## Features & Rationale
### Recommendation System
1. Product Data
- Product Name
- Description
- Ingredients

2. User Data
- Concerns
- Skin Type
- Age Group
- Past Purchase

### Buy Likelihood
1. Sessions (past 7 days)
2. Average Duration
3. Products Viewed
4. Cart Adds


## Techniques
### Recommendation System
#### Content-Based Filtering
Content-based filtering recommends items similar to other items in the past. For example, if someone likes dolls, the system also recommends those that have similar feature values to dolls. Dolls are categorized as women's goods, so women's goods will also be recommended to users who buy dolls. What has just been mentioned is just an example, it can also involve other features to create the recommendation system.

The advantage of this system is that it can be used for new customers. This is because new customer data usually does not have rating data. Therefore content-based filtering can be run with very little data that has been entered in the sample. While the drawback of this measurement is that it is not easy to carry out metric measurements.

The principle of this system is to measure the similarity score of each item with every other item, and then order the similarity scores. The items with the highest similarity scores will be displayed as recommendation results.

For input processing, if there is a review in the form of a sentence, then TfIdfVectorizer or CountVectorizer is used. And if there are categorical features, one-hot encoding is used. Furthermore, the similarity measurement uses several algorithms, one of which uses cosine_similarity.

#### Collaborative Filtering
Collaborative filtering recommends items based on user ratings of items. This system is divided into memory-based and model-based. Memory-based itself is further divided into user-based and item-based. Meanwhile, model-based is divided into cluster-based, matrix factorization and deep learning.

User-based searches for users with similar characteristics, for example, they both have dry skin type, or they both concern about aging and wrinkles. Item-based calculates the similarity between each item from all users. Cluster-based uses a clustering algorithm (K-Means, Gaussian Mixture Model, DBScan) to group users according to available variables. Matrix Factorization uses matrix decomposition formation. Meanwhile systems with deep learning use deep learning algorithms to determine the final recommendation.

In the product recommendation system approach by collaborative filtering using the embedding technique, this time user profile, item and user ratings will be used for the item. Each user and item will be embedded to be modeled with the target, namely the rating itself.

The advantage of collaborative filtering is that it is not difficult to measure metrics. While the drawback is that it cannot be used if there is no user rating/rating data for the item.

### Buy Likelihood

## Evaluation
### Recommendation System
#### Content-Based Filtering
The content-based filtering evaluation uses [precision@k](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54) where the precision of only k recommendations is measured which are issued. Precision@k is formulated with:

```
precision@k = (# of the relevant @k recommended items) / (# @k recommended items)
```

The definition of a relevant item is an item that has a cosine_similarity value greater than 0.7. As an example of its application, item 2360 has a precision@k of 100% because all values are above the threshold. Here's an implementation of the code.

```
item_id = pd.Series(f.index).sample(1).iloc[0]

def recommendations(item_id=item_id, threshold=0.7, n_item=10):
  print(f'Showing similar item for item {item_id}')
  print()
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1
    print('Item', i, '|', 'Cosine Similarity Value :', j)
  print()

  c = b*100/n_item
  print('Precision(%):')
  return c
  
recommendations(item_id)
```

This is the example of calculating cosine similarity per item with another item.

![content_based_2360_2](https://github.com/alvinrach/Product-Recommender-System/blob/main/content_based_2360_2.png?raw=true)

Here repeated measurements were also carried out with 100 arbitrary items, with one output item as many as 10 pieces with the same threshold (0.7). The measurement results are then averaged. Measurements show the system has a precision@k of 99.8%. Here's an implementation of the code.

```
def _evalRecommendations(item_id=item_id, threshold=0.7, n_item=10):
  a = np.argsort(cosine_sim_df[item_id].values)
  a = a[~np.isin(a,item_id)]
  a = a[-n_item:][::-1]
  a = {i:cosine_sim_df[item_id][i] for i in a}
  a

  b=0
  for i,j in a.items():
    if j>threshold:
      b+=1

  c = b*100/n_item
  return c

def evalRecommendations(n_to_eval=100):
  a = 0
  for i in range(n_to_eval):
    item_id = pd.Series(f.index).sample(1).iloc[0]
    a = a + _evalRecommendations(item_id)

  print(f'Evaluation (Precision) for {n_to_eval} times try is : {a/n_to_eval} %')

evalRecommendations(100)
```

![eval_avg_content_based](https://github.com/alvinrach/Product-Recommender-System/blob/main/eval_avg_content_based.png?raw=true)

The advantage of the precision@k metric is that it is practical, there is no need to compute the entire dataset. The disadvantage of precision@k is that it is sometimes inconsistent / varied, because it only counts on a number of k recommendations issued. However, this variation can be reduced by taking the average as mentioned above.

#### Collaborative Filtering
To evaluate, the metric used is Root Mean Squared Error.
- Root Mean Squared Error (RMSE) is one of the metrics used in the case of regression, in addition to Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE). The formula for this metric is formulated as follows:

![RMSE Formula](https://miro.medium.com/max/412/1*RSYTYpqyGDYWPmI0rD8zqA.png)

Where n is the number of samples, y is the actual value, and is the predicted value.
- The advantage of RMSE over other regression metrics is that it is suitable if you want to take into account the error value of a larger outlier, if that is so desired. The disadvantages of RMSE compared to other regression metrics are that it is less easy to interpret (compared to MAE) and makes the metric too sensitive in calculating the error value of outliers.
- This is the example of RootMeanSquaredError calculation (compiling) in Tensorflow/Keras:
```
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer = keras.optimizers.Adam(learning_rate=0.0005),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
```
If neural network is used, then the training will be done each epoch. Here is the example of training in deep learning case. It is expected that validation loss will be reduced each epoch, along with training. Reduced training loss without validation loss reduction can hint overfitting problem. While bad training and validation means underfitting. In this case we manage to reach a good fit.

![plot_rmse_recommender](https://github.com/alvinrach/Product-Recommender-System/blob/main/plot_rmse_recommender.png?raw=true)

### Buy Likelihood

## Business Impact

## Deploy

### Prototyping

#### Recommendation System
![Recommender Prototype](https://github.com/alvinrach/recommendation-buy-likelihood/blob/main/images/recommender.png?raw=true)

#### Buy Likelihood
![Buy Likelihood Prototype](https://github.com/alvinrach/recommendation-buy-likelihood/blob/main/images/buy-likelihood.png?raw=true)

### Deployment

#### ML model deployment
1. Model Deployment Strategy

- Deployment pattern - REST API, batch inference, or real-time streaming  
- Model versioning - A/B testing between model versions, rollback procedures  
- Environment promotion - Model validation gates before production  

2. Model Monitoring & Observability

- Performance metrics - Accuracy, latency, throughput in production  
- Data drift detection - Input data distribution changes over time  
- Model degradation alerts - Automated alerts when performance drops  

3. Infrastructure & Integration

- Scaling strategy - Handle varying inference loads (auto-scaling, caching)  
- Data pipeline integration - How training/inference data flows through systems  
- Fallback mechanisms - What happens if model fails (default rules, previous model version)  

4. CI/CD Pipeline (& MLOps)

- Model training pipeline - Automated retraining triggers and validation  
- Model deployment automation - Automated testing and promotion to production  
- Version control - Model artifacts, code, and configuration management  

#### Data Pipeline

1. Pipeline Orchestration & Scheduling

- Workflow management - How jobs are scheduled, dependencies managed, retry logic  
- Data processing patterns - Batch vs streaming, micro-batching intervals  
- Error handling - Failed job recovery, data quality validation gates  

2. Data Quality & Monitoring

- Data validation - Schema validation, data completeness checks, anomaly detection  
- Pipeline observability - Job success rates, processing times, data lineage tracking  
- Alerting - Data freshness alerts, volume anomalies, processing failures  

3. Scalability & Resource Management

- Elastic processing - Auto-scaling compute resources based on data volume  
- Storage optimization - Data partitioning, compression, archival strategies  
- Performance tuning - Bottleneck identification, parallelization, resource allocation  