## Churn Prediction

In this chapter you will learn churn prediction fundamentals, then fit logistic regression and decision tree models to predict churn. Finally, you will explore the results and extract insights on what are the drivers of the churn.

### What is churn?
Churn definition will depend on the company and its business model, but essentially churn event happens when a customer stops buying a product, using a service, or engaging with a product or application. Churn can happen in either a contractual or non-contractual business context. Contractual churn happens when customers explicitly cancel a service or subscription, while non-contractual is harder to observe and requires in-depth data exploration. Also, churn can be viewed as either voluntary or involuntary. Voluntary churn means customers decided to stop using the product or a service, while involuntary churn happens when customers fail to automatically update their subscription due to credit card expiration or other blockers.

### Types of churn
The churn definition is different for contractual and non-contractual business models. Contractual churn happens explicitly - when customers decide to cancel their subscription or service. Non-contractual churn happens in settings like grocery or online shopping where customers just stop buying or using the product without explicit termination of a contract.

### Modeling different types of churn
Typically, non-contractual churn is harder to define and predict, as the company has to track their customer purchase patterns and define churn with certain rules, for example, churn might mean no grocery activity in 1 month. Defining this period is both art and science as different customers have different purchasing patterns, inter-purchase frequencies and so on. Modeling this type of churn is not in the scope of this course. In this example, we will model contractual churn in the telecom business model, where customers can have multiple services with a telecommunications company under one master agreement which defines whether customer is still active, or has churned, which means they have terminated their contract.

### Encoding churn
Churn is typically encoded with ones and zeros, one meaning churned and zero for no churn. Different datasets could have strings or other values for churn, but the best practice is to encode them into ones and zeros. In our telecom dataset you will see that that the Churn column has ones and zeros in it.

### Exploring churn distribution
One thing that's important to explore is whether there is a severe class imbalance meaning there are large differences in the number of observations in each class. In this case we can see that there are over 26% churned customers and over 73% non churned customers. There is some imbalance, but not a severe one. Typically if the minority class is less than 5% then we should worry and explore computational ways to increase the minority class or decrease the majority class with oversampling or undersampling techniques.

### Split to training and testing data
One of the important steps that we quickly explored in chapter 1, was data splitting into training and testing. We need to do this to make sure our model generalizes and can perform well in predicting unseen data. This can be done by running train_test_split function from sklearn library. The test_size parameter expects a value between 0 and 1 that defines the percentage of observations to be randomly assigned to the testing dataset.

### Separate features and target variables
Once the training and testing split is completed, there's one final thing to do before we move to modeling. In this step, we separate the independent features and the target variable. We do this by encoding our unique identifier for the customers as a list called custid. Then, our target variable as a list called target. And then we build a list of independent feature names called cols using a list comprehension to iterate through column names and exclude the target and customer id columns we have just defined. Finally, we use the previously built train and test datasets, and create features and target datasets for both training and testing.

## Predict Churn with Logistic
Logistic regression is a supervised learning technique that predicts binary response variables. Logistic regression models the logarithm of the odds ratio. Odds is a ratio of the probability of the event occurring divided by the probability of the event not occurring, or p divided by 1 minus p. For example, if the probability of the churn is 75% then probability of no churn is 25%, hence the odds ratio will be 75% divided by 25%, or 3. The logarithm of 3 is roughly 0.48. The reasons behind the math are beyond the scope of this course, but this approach helps to find the decision boundary between the two classes while keeping the coefficients linearly related to the target variable. Here's the formula of the logistic regression equation based on two input variables and a probability p.

### Modeling steps
As we've seen before, supervised learning has five steps: Split the data into training and testing. Initialize the model. Fit the model on the training dataset. Then, predict the values on the testing data. And finally, evaluate the model performance by comparing the predicted values with the actual ones in the testing data. Since we have already learned and tested how to split the data to training and testing, we will now move to the second step.

### Fitting the model
Let's fit the model. First, we import the logistic regression classifier from scikit learn library. Then, we initialize the model instance. Finally, we fit the model on the training data by providing the input features and the target variable to the method called "fit".

### Model performance metrics
Once the model is fitted, we are ready to assess its performance. In the previous lesson, we looked at high level accuracy metrics, but there are more. While this is not an exhaustive list, these are good metrics to start with, and they are easy to interpret. The one we used previously, is accuracy, which is the percentage of correctly predicted values compared to the actual ones. This includes prediction accuracy for both classes combined - both how many churned and non-churned customers we have correctly labeled. This gives us the main performance of the model irrespective of the class. The other one is called precision, and it measures the number of positive class predictions that were correct. In this case this is the number of customers that were predicted as churned, and did actually churn. The third metric is called recall. It measures the number of total positive class observations that were correctly captured. In this case it is the number of total churned customers that were correctly classified as such.

### Measuring model accuracy
Now, let's move to calculating the model accuracy on both training and testing dataset. Typically, the testing metric should be lower, as these are the unseen observations, while the training data was used to train the model. First we import the accuracy_score from sklearn.metrics module. Then we predict the labels calling the predict method on the logistic regression instance and passing the input features. Once completed, we call the accuracy score and feed the actual labels first, and the predicted ones afterwards. We store the accuracy scores as separate objects. Finally, we print the rounded accuracy. We can see that the training accuracy is around 81%, while the testing is roughly 80%. This means we have correctly labeled 80% of the customer churn events.

### Measuring precision and recall
Now, let's calculate precision and recall. The steps are identical to calculating accuracy. First we import the functions. Then, we calculate the precision score for both training and testing data, and round it to 4 decimals. And the same for the recall score. Finally, we print them out. As we can see, these values vary a bit more between training and testing. Also, they lower than accuracy which means the model predicts the minority churn class less accurately than the majority non churned class.

### Regularization
Now, we'll learn the concept of regularization. The main idea is to introduce a penalty coefficient for model complexity in the model fitting phase. The penalty addresses over-fitting which occurs when we have too many features. In that situation, the model just memorizes the patterns in the training data, but does not predict well on the testing data. Some regularization techniques like L1 also perform feature selection which reduces the number of inputs in the model, simplifies it, and makes it more generalizable to unseen samples.

### L1 regularization and feature selection
Let's test the regularization now. Logistic Regression from sklearn already performs regularization by default. It is L2 or ridge regularization which only manages over-fitting but does not perform feature selection. L1 regularization, also called LASSO, can be called explicitly. This approach performs feature selection by shrinking some of the beta parameters to zero. We can call it by providing 'l1' to penalty argument, C value which is the inverse of the regularization strength - more on this later. Finally, we feed the 'liblinear' as solver that will be used for L1 regularization. Then, we fit the data as previously. Now, what should be the C value? We will have to optimize the C value by tuning it.

### Tuning L1 regularization
We will list a number of different C values and build a model for each. Typically we explore C between 0 and 1 although values greater than 1 are also acceptable. Then, we create an empty numpy array with zeros, and add C candidates in the first column. Afterwards, we iterate through the C values, and build logistic regression with each. Then, we store the count of non-zero coefficients, accuracy, precision and recall in the remaining columns, and finally print it to investigate.

### Choosing optimal C value
We can see that lower C values shrink the number of non-zero coefficients, while also impacting the performance metrics. The decision on which C value to choose depends on the cost of declining precision and / or recall. Typically, we would like to choose a model with reduced complexity that still maintains similar performance metrics.

### Choosing optimal C value
In this case, C value of 0.025 meets this criteria - it reduces the number of features to 13, while maintaining the accuracy, precision and recall scores close to the ones in the non-regularized model. The other models with lower C values start experiencing decline in the recall metric.

## Predict Churn with Decision Trees
Here, we have an example decision tree that was built on a famous Titanic survival dataset. The decision tree outlines the if-else rules that were inferred from the survival dataset. We can see that the survival probabilities differ for each of the leaves depending on the rules.

### Modeling steps
To cement our knowledge, we'll go through the supervised learning modeling steps again: Split the data into training and testing. Initialize the model. Fit the model on the training dataset. Then, predict the values on the testing data. And finally, evaluate the model performance by comparing the predicted values with the actual ones in the testing data. Since we have already learned and tested how to split the data to training and testing, we will now move to the second step with the decision trees.

### Fitting the model
Now, we will fit the model. First, we import the classifier from scikit learn library. Then, we initialize the decision tree instance. Finally, we fit the model on the training data by first providing the input features and then the target variable.

### Measuring model accuracy
Now, let's move to calculating the model accuracy on both training and testing datasets. As with the logistic regression, the steps are the same. First, we import the accuracy_score from sklearn.metrics module. Then we predict the labels calling the predict method on the fitted tree instance. Once completed, we call the accuracy_score and feed the actual labels first, and the predicted ones afterwards. We store the accuracy scores in separate objects. Finally, we print the rounded accuracy, and can see that the training accuracy is around 99.7%, while testing accuracy is only 72%. This is different from logistic regression where both numbers were similar around 80%. This indicates that the tree memorized the patterns and rules for the training data almost perfectly, but failed to generalize the rules for the testing data. We will learn how to reduce the size of the tree to manage this in the next slides.

### Measuring precision and recall
Now, let's calculate the precision and recall. The process is identical to the one we did with logistic regression. First, we import the functions to calculate precision and recall. Then, we calculate the precision score for both training and testing data, round it to 4 decimals, then the same for recall score. Finally, we print the numbers. One thing that stands out, is the low value for testing recall, while the other scores are over 99%. Remembering that recall means the number of total churned instances correctly captured by the model, we can see that the model is very precise with its prediction, but fails to identify more than half of the actually churned customers.

### Tree depth parameter tuning
The decision tree is very prone to over-fitting as it will build rules that will memorize all the patterns down to each observation level. To manage this, we need to prune the tree, which means limiting the number of if-else rules. To do this, we need to provide max_depth parameter. We will tune it in the same way we tuned the C value for logistic regression. First, we create a list of max_depth candidates between 2 and 14, then create a numpy array with zeros, and store the depth candidates in the first column. Then we iterate through the depth values, and fit a decision tree for each. Afterwards, we calculate the accuracy, precision and recall scores on the testing data and store them into the numpy array. Finally, we print the results as pandas DataFrame for better formatting.

### Choosing optimal depth
As we can see, the testing accuracy first increases with more depth and then starts to decline. The precision declines with more depth, yet the recall increases first, then starts falling.

### Choosing optimal depth
We can see that at the max_depth of 5, the tree solution produces good scores, and a pretty high recall metric before it starts declining. This should be the starting point for the first choice of the model.

## Identify and Interpret Churn Drivers

### Plotting decision tree rules
Decision trees are neat because they just are list of nested if-else rules that can be plotted. Here's an example of a 3-level decision tree. To plot it, we have to import the tree module from sklearn, and the graphviz module which needs to be installed separately if you want to use it on your own machine. Then, we export a graphviz object by passing the fitted decision tree object, column names, precision level, class names, and whether to fill the leaves with color. Finally, we call the Source method from the graphviz module, pass the exported object, and then call the display function on it.

### Interpreting decision tree chart
The result is a good looking decision tree visualization. You can interpret it as a set of if-else rules starting from the top. The first row in each leaf is the rule that is then branched by whether or not it is met. We can see the True and False labels on the arrows flowing from the parent leaf to the child leaves. We can see that customer tenure is the most important variable. If the tenure is lower than 11.5, and the customer has no fiber optic Internet service, then it is very likely that customer will churn. The tree can be built with more layers, and this will give more insight into other variables driving churn.

### Logistic regression coefficients
Now, with logistic regression we get coefficients. The coefficients can be interpreted as the change in log-odds of the churn associated with 1 unit increase in the input feature value. For example if the input feature is tenure in years, then increase in the tenure by one year will have an effect equal to the coefficient to the log-odds. Here's the formula outlining the model equation. What's the main challenge? Log of odds is incredibly hard to interpret.

### Extracting logistic regression coefficients
Before we begin with interpretation, let's see how to extract the coefficients. We can use the coef underscore method on the fitted logistic regression instance. With that, we get a list of beta coefficients for each input variable.

### Transforming logistic regression coefficients
The challenge with the previous list is two-fold: First, the coefficient values come without names, and second, the coefficients are in the log-odds scale which is difficult to interpret. The solution is to calculate the exponent of the coefficients, which will give us the change in the actual odds associated with 1 unit increase in the feature value. In Python, we first want to build a pandas dataframe with columns names and coefficients, and name them accordingly. Once that is done, we calculate the exponent of the coefficients, and store them in a separate column. Once that is completed, we extract the non-zero coefficients and print them sorted by the largest Coefficient values first.

### Meaning of transformed coefficients
Then we get this view. It does not account for statistical significance which we will assess in the next chapter. We can see that the feature with the largest effect on the odds of churning is tenure which is consistent with the findings from the decision tree. The interpretation of the coefficient for odds is as follows - values less than 1 decrease the odds, and values more than 1 increase the odds. The effect on the odds is calculated by multiplying the exponent of the coefficient. So the effect of one additional year of tenure decreases the odds of churn by 1 minus 0.403. This translates to roughly 60% decrease in the churn odds.
