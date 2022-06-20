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
