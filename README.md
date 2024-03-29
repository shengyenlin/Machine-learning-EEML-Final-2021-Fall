# Machine learning - final project: G-Research Crypto Competition

## Competition description

In this competition, you'll use your machine learning expertise to forecast short term returns in 14 popular cryptocurrencies. We have amassed a dataset of millions of rows of high-frequency market data dating back to 2018 which you can use to build your model.

## Competition result

[Kaggle](https://www.kaggle.com/competitions/g-research-crypto-forecasting/overview): 1043/1947

We used LightGBM and data leak with following feature engineering techniques for our final model. Please refer to `lightgbm-all.ipynb` for codes and details.

## Feature engineering
- Please refer to `DataPreprocessing.py`, `FeatureEngineering_tutorial.ipynb`(storing the results)
- feature engineering steps
    - load data
    - preprocess for timestep
    - whehter to load strict data or not
    - slice by asset id
    - fill missing timestamp (sue forward filling from tutorial notebook, may try different methods in experiment)
    - create different technical analysis features with different parameter sets

## Feature selection
- Please refer to `FeatureSelection.py`, `FeatureEngineering_tutorial.ipynb`(storing the results)
- feature selection steps
    - L1 regularization regression
        - tunable params: nubmer of cv, max_iter (won't change result too much)
    - correlation coefficients 
        - tunable params: threshold on correlation, (or may just set a upper bound on number of selected features) 
    - union features selected by the two above methods

## Useful Links
- project drive: https://drive.google.com/drive/folders/13ueTjXCT1uD0yghKgkTdWN6uVQ0Ru99P?usp=sharing
- Kaggle tutorial: https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition
- feature engineering md: https://hackmd.io/wurSzyc5RyKJ_tiJnWQMpg
- report/presentation structure: https://docs.google.com/document/d/1407GC_CDIjMhYRu-AsMAwVK2ShO7OAlKwXhhdTO9S34/edit?usp=sharing
- 0102 meeting notes: https://docs.google.com/document/d/1W8sWToLrOlfixD2aWEyJPW4ww1qCw0LfdGHJo1a4cUs/edit?usp=sharing
