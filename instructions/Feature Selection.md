**Feature Selection**  
Why do we need feature selection?
1. Curse of dimensionality. The more features the more configurations (increase exponentionally).
2. Keep the model simple and explainable.
3. Excluding non-informative features, otherwise Garbage in, Garbage out.

The methods:
1. Pearson correlation
2. Chi-square test
3. Recursive feature elimination
4. Lasso. Lasso norm regularization drives parameters to zero.
5. Tree-based

Sklearn has module for feature selection:  
`from sklearn.feature_selection import SelectFromModel`

**References**  
1. [The 5 Feature Selection Algorithms every Data Scientist should know](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2)