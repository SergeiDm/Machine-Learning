**Feature engineering**
1. Encoding categorical variable
* Label encoding - normalize labels such that they contain only values between 0 and n_classes-1.
* One-hot encoding
* Embeddings
* Mean encoding - ratio of occurrence of the positive class in the target variable. It represents probability of your target variable, conditional on each value of the feature.  
`Mean_encoded_subject = df.groupby(['SubjectName'])['Target'].mean().to_dict()`  
`df['SubjectName'] =  df['SubjectName'].map(Mean_encoded_subject)`  
    Pros:  
    - Increase feature predictive power
    - Creates a monotonic relationship between the variable and the target  
    
    Cons:  
    - Information leakage and as a result overfitting  

GBT can not work properly with categorical variable with high cardinality.