# ML-Helper
Test your performance using several Machine Learning Algorithms for your dataset.
# Description
Test many machine learning algorithms for your dataset with much less code. Also it's contains some visualize methods for your dataset. For example, feature importances HBar Plot, Train-Test Accuracy comparison Plot.
This python file has five classes. These are;
MLAlgorithmsComparison,
ClassificationAlgorithmsOptimizer,
RegressionAlgorithmsOptimizer,
OtherRegressionAlgorithmsOptimizer,
SomeMethods.
We'll use only four classes. These are;
MLAlgorithmsComparison,
ClassificationAlgorithmsOptimizer,
RegressionAlgorithmsOptimizer,
OtherRegressionAlgorithmsOptimizer.
# MLAlgorithmsComparison
With the object you will create from this class, you can find optimal ML Algorithm for your dataset. 
If you want to find a solve for Classification Problem, you must choose classificationAlgorithms method.
If you want to find a solve for Regression Problem, you must choose regressionAlgorithms method.
If you want to use Ridge, Lasso or ElasicNet Algorithms on your dataset, you must choose otherRegressionAlgorithms method.
# ClassificationAlgorithmsOptimizer
After doing the MLAlgorithmsComparison operation, you can make Hyper Parameter Optimization for some algorithms on your classification problem if you want. Of course, keep in mind that this will be at a basic level.
# RegressionAlgorithmsOptimizer
Same as above but this time for your Regression problem.
# OtherRegressionAlgorithmsOptimizer
For other regression algorithms.

# Usage
You must slice on your dataset as features and label(This python file represents them as X and y). Once you give the parameters the required values, you are ready to use it.
# dimensionReduction
If you want to test your dataset with Dimension Reduction, you must give true to dimensionReduction parameter.
