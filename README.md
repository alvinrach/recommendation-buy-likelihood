## Data Sources & Acquisition Strategy
### Primary Data Source: Internal Platform
Customer transaction histories, browsing behaviors, and product interactions are captured through the existing e-commerce infrastructure. User profiles containing skin types, concerns, and demographic information are collected during registration. Product catalogs with ingredients, pricing, and ratings are maintained in the current inventory system.

### External Data Sources (If Required)
**Web crawling** will be implemented to collect supplementary data from established beauty retailers while ensuring compliance with robots.txt and terms of service. **Kaggle datasets** such as "Sephora Products and Skincare Reviews" (8K+ products, 1M+ reviews) and "E-Commerce User Behavior Data" (285M+ user events) will provide additional training data for model development and benchmarking.

### Data Integration
Standardized schemas and validation processes will ensure data quality when combining internal and external sources. Privacy compliance will be maintained through anonymization and adherence to local data protection regulations.

## Features & Rationale
### Recommendation System Features
1. Product Data
- Product Name & Description: Essential identifiers that enable content-based filtering through text similarity analysis. Product descriptions are processed using NLP techniques to extract key attributes and match them with user preferences and concerns.
- Ingredients: Critical for skincare recommendations as ingredient compatibility directly impacts user satisfaction and safety. Ingredient vectors are created to measure similarity between products and match against user's skin sensitivities or preferences (e.g., users preferring retinol-based products).
- Price: Enables price-based filtering to respect user budget constraints and spending patterns. Historical purchase data reveals price sensitivity, allowing recommendations within appropriate price ranges to improve conversion likelihood.
- Rating: Quality indicator that influences recommendation ranking. High-rated products are prioritized to ensure user satisfaction, while rating patterns help identify trending or trusted products within specific categories.

2. User Data
- Concerns: Primary driver for personalized recommendations as skincare needs vary significantly (acne, aging, hyperpigmentation). Concern-based filtering ensures recommended products address specific user problems, improving relevance and purchase intent.
- Skin Type: Fundamental compatibility filter preventing recommendations of unsuitable products (e.g., oil-based products for oily skin). This hard constraint ensures user safety and satisfaction while reducing return rates.
- Age Group: Influences product suitability as skincare needs evolve with age. Younger users may prefer acne-focused products while mature users seek anti-aging solutions, enabling age-appropriate recommendations.
- Past Purchase: Reveals user preferences, brand loyalty, and category exploration patterns. Purchase history enables collaborative filtering and identifies complementary products for cross-selling opportunities.

### Purchase Likelihood Features
1. Behavioral Features
- Page views (7/14/30 days): Indicates engagement level and purchase intent. Higher page view frequency suggests active product research, while declining views may indicate lost interest requiring re-engagement campaigns.
- Time spent on product pages: Quality engagement metric showing genuine interest versus casual browsing. Extended time on specific products indicates serious consideration and higher purchase probability.
- Cart additions/removals: Strong purchase intent signals, with cart additions showing immediate buying interest. Cart removal patterns help identify price sensitivity or product comparison behaviors.
- Search queries and patterns: Reveals user intent and specific needs. Specific ingredient searches or problem-focused queries indicate higher purchase likelihood than general browsing patterns.
- Session frequency and duration: Overall engagement indicators showing user loyalty and platform familiarity. Frequent, longer sessions suggest higher lifetime value and purchase probability.

2. Temporal Features
- Days since last visit: Recency indicator affecting purchase likelihood. Recent visitors show higher engagement, while extended absence may indicate churn risk requiring targeted re-engagement efforts.
- Seasonality patterns: Captures cyclical purchasing behaviors influenced by weather, holidays, or personal routines. Seasonal adjustments improve prediction accuracy during peak buying periods.
- Purchase cycle patterns: Individual user purchasing rhythms based on product consumption rates. Understanding personal replenishment cycles enables proactive targeting before product depletion.

3. Product Interaction Features
- Categories viewed: Shows breadth of interest and routine completeness. Users exploring multiple categories (cleanser, serum, moisturizer) indicate comprehensive routine building with higher basket values.
- Price ranges browsed: Reveals budget constraints and willingness to pay premium prices. Price browsing patterns help predict purchase likelihood within specific price segments.
- Product ratings viewed: Research behavior indicating purchase consideration. Users checking ratings and reviews demonstrate serious buying intent and quality consciousness.
- Reviews read: Deep engagement signal showing thorough product evaluation. Review reading patterns correlate with purchase decision-making and reduce post-purchase regret likelihood.

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
Traditional Machine Learning:
1. Logistic Regression:
Interpretable coefficients are provided, making it suitable as a baseline model. Feature relationships can be easily understood by business stakeholders.
2. Random Forest:
Feature interactions are automatically handled through ensemble methods. Feature importance scores are generated to identify key predictive behaviors.
3. Gradient Boosting (XGBoost/LightGBM):
Superior performance on tabular data is typically achieved. Complex behavioral patterns are captured while overfitting is prevented through cross-validation.
4. Support Vector Machines:
High-dimensional feature spaces are effectively handled through kernel transformations. Non-linear decision boundaries are created for complex user behavior patterns.

Deep Learning:
1. Neural Networks:
Complex non-linear relationships are automatically learned through multiple layers. Feature engineering requirements are reduced as patterns are discovered in raw data.
2. LSTM/GRU:
Sequential user behavior patterns are processed through recurrent architectures. Time-dependent relationships in browsing and purchase sequences are captured.

Based on the e-commerce purchase prediction requirements, Gradient Boosting (XGBoost/LightGBM) is recommended as the primary approach. This choice is justified because superior performance on tabular behavioral data is consistently achieved, while interpretability is maintained through feature importance rankings that business stakeholders can understand. The inherent class imbalance in e-commerce conversion rates (typically 2-5%) is effectively handled through built-in class weighting parameters, and robust performance is ensured through cross-validation techniques. Additionally, the model's ability to capture complex feature interactions between user behaviors (session patterns, product views, cart activities) without requiring extensive feature engineering makes it well-suited for production deployment where new behavioral signals may emerge over time. However, extensive experimentation will be conducted across multiple algorithms to ensure optimal performance.

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

1. Technical Performance Metrics:
AUC-ROC will be calculated to measure the model's ability to distinguish between purchasers and non-purchasers across all classification thresholds. AUC-PR (Precision-Recall) will be prioritized over AUC-ROC due to the severe class imbalance inherent in e-commerce conversion rates, as it better reflects performance on the minority class. F1-Score will be computed to balance precision and recall trade-offs, while precision and recall will be evaluated at business-relevant thresholds to optimize for specific campaign targeting requirements.
2. Business-Oriented Metrics:
Lift analysis will be conducted to quantify improvement over random targeting, with measurements taken at different percentile cuts (top 10%, 20%, 30% of predicted users). Conversion rate within the top-scored segments will be calculated to directly translate model performance into expected campaign effectiveness. Customer Acquisition Cost (CAC) reduction will be measured by comparing marketing efficiency before and after model implementation, while Return on Investment (ROI) will be calculated based on increased revenue from improved targeting accuracy.
3. Validation Strategy:
Time-based cross-validation will be implemented to prevent data leakage, where models are trained on historical periods and tested on future time windows. Stratified sampling will maintain class distribution across validation folds, ensuring representative evaluation despite class imbalance. Walk-forward validation will simulate real-world deployment scenarios, with model performance monitored across multiple time periods to assess stability and generalization capability.

## Business Impact
### Recommendation System Impact
**Revenue Growth** of 15-25% is expected through improved product discovery and personalized user experiences. **Average Order Value (AOV)** will increase by 10-20% as users discover complementary products within their skincare routines. **Customer Retention** will improve by 20-30% through more relevant product suggestions that better address individual skin concerns and preferences.

### Purchase Prediction Impact
**Marketing Efficiency** will improve by 30-40% through targeted campaigns focused on high-intent users, reducing wasted advertising spend on low-probability prospects. **Customer Acquisition Cost (CAC)** will decrease by 20-25% as marketing resources are allocated more effectively. **Conversion Rates** will increase by 15-30% when promotional campaigns target users with highest purchase likelihood scores.

### Combined Operational Benefits
**Inventory Management** will be optimized through better demand forecasting based on user intent predictions and recommendation patterns. **Customer Support** workload will decrease by 15-20% as more accurate recommendations reduce product returns and compatibility issues. **Competitive Advantage** will be established through personalized experiences that increase customer loyalty and platform stickiness in the crowded skincare e-commerce market.

### Long-term Strategic Value
**Data-Driven Culture** will be fostered throughout the organization as success metrics demonstrate the value of machine learning initiatives. **Platform Scalability** will be enhanced as recommendation and prediction systems can be extended to new product categories and user segments with minimal additional investment.

## Deploy
### Prototyping
#### Recommendation System
Here is the prototype for recommendation system in React. For example, user_002 has dry type skin, aging and wrinkles concerns, 35-44 age group and 2 past purchases. Here in the planned dashboard we will show several related products to this user, ordered by match score.

![Recommender Prototype](https://github.com/alvinrach/recommendation-buy-likelihood/blob/main/images/recommender.png?raw=true)

#### Buy Likelihood
And this is the prototype of buy likelihood dashboard in React. We have features that we already process, such as sessions, average duration, products viewed and cart adds. In the label or target area we have purchase probability and intent segment, which are:

🟢 Green: High probability (>60%) = High Intent = Good for business
🟡 Yellow: Medium probability (30-60%) = Medium Intent = Moderate
🔴 Red: Low probability (<30%) = Low Intent = Needs attention

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