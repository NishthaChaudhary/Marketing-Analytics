## Customer Lifetime Value Prediction
In this chapter, we will explore the Customer Lifetime Value concept, and different ways to calculate it. Then, we will learn how to use regression to predict customer transactions in the next month.

### What is CLV?
The customer lifetime value is a measurement of how much a company expects to earn from an average customer in a lifetime. It can be historical - where we sum up each customer's profit and get the actual customer value. Another alternative is to predict the customer lifetime value which is a bit more complex. There are a number of methods to predict customer lifetime value. We will not delve into the complex prediction methods like Pareto / NBD or beta-geometric / NBD models which would take a full course of their own. Instead, we will use simple traditional methods to get decent estimates. Also, we will learn how to predict continuous variables like number of purchases with regression.

### Historical CLV
Historical customer lifetime value is simply a sum of the revenues of customer transactions, multiplied by the average or product-level profit margin. Alternatively, we can sum up the actual profit from each purchase, if that data is available. The main challenge with the historical approach is that it does not account for customer tenure, retention and churn rates. If the company is acquiring a lot of new customers - their average historical lifetime value will be deflated due to their short tenure. The second challenge is that the new customers are treated the same way as tenured customers, therefore the average customer lifetime value does not represent the potential future revenue.

### Basic CLV formula
The most basic lifetime value calculation is to take the average revenue per customer within a certain period, let's say a month, multiply it by the profit margin, and then multiply the result by the average or expected customer lifespan. The lifespan can be defined using company's knowledge about its customers, or analyzing the average time it takes for customers to churn.

### Granular CLV formula
A more granular version of the basic customer lifetime value formula looks at each transaction. Here, we multiply the average revenue per purchase or transaction with the average frequency within a defined period - for example a month - and then multiply that with the profit margin. Afterwards, we multiply the result with the average customer lifespan. This method accounts for the frequency of the transactions within a certain timeframe, as well as the average revenue per transaction, therefore capturing more granular data points. Still, it does not account for customer retention rates, and assumes the frequency and revenue per transaction will stay the same within the defined lifespan.

### Traditional CLV formula
The traditional formula is the most popular descriptive customer lifetime value technique. It incorporates retention and churn rates. We calculate it by multiplying the average revenue with the profit margin, and then with the ratio of retention to churn. Churn is defined as 1 minus retention. The retention to churn ratio gives us a multiplier, that acts as a proxy to expected length of the customer lifespan with the company. It is useful, but assumes that the churn is final, therefore the timeframe used for the retention and churn is critical. Especially, in the non-contractual business setting, we need to make sure customers who are defined as churned within this timeframe, don't actually come back later.

### Introduction to transactions dataset
We will work with an open source online retail dataset. It has transactions from a retailer with variables on money spent, quantity and other values for each transaction. This is a standard transactional dataset.

### Introduction to cohorts dataset
We will also use the cohorts dataset that is derived from the online retail data. This dataset is created by assigning each customer to a monthly cohort, based on the month they made their first purchase. Then a pivot table is built with the activity counts for each cohort in the subsequent months. We will use this dataset to calculate their retention rates. If you want to see how this dataset is calculated, you can check the Customer Segmentation in Python course where we build this and the other datasets from scratch.

### Calculate monthly retention
The cohort dataset already has monthly active users for each monthly cohort. This means that in each row we have the same group of customers who started buying on month one, and then some of them come back in the subsequent months, while some don't. To calculate retention, we first build a cohort sizes dataset, which is just the first column from the cohort counts data. Then, we calculate the retention by dividing the cohort counts by the cohort sizes. We can also calculate customer churn rates, which is 1 minus retention. Finally, we can plot the newly built retention table using heatmap function from seaborn package.

### Retention table
As you can see, the first month retention is 100%. This is because this is the month when the customers had first started buying. And by definition this means all of the customers from this cohort are active in their first month.

### The goal of CLV
Let's reiterate first on the goals of customer lifetime value. With CLV we want to measure customer value in terms of revenue or profit. This way we benchmark the customers and are able to assess the maximum amount of money the company can afford to spend in acquiring new customers, given the lifetime value they are expecting to earn from an average customer. In our example dataset, we don't have the profit margin and we won't assume an artificial margin. For the sake of simplicity we will skip the profit margin from the calculation and will calculate revenue-based CLV. Here is the traditional CLV formula update to revenue-based form. We will use the revenue-based methodology in all three methods.

### Basic CLV calculation
Great, let's start with the basic customer lifetime calculation. First, we calculate the revenue spend for each customer monthly. We group by the CustomerID and the InvoiceMonth, and then sum up the revenue stored in the TotalSum column, and then calculate the overall average. Afterwards, we define the customer lifespan. This is a broad topic that could take a full course on its own, but ultimately this depends on the business model, customer lifetime expectation, and other data points. This can be inferred by looking into the average time it takes customers to churn from the time they made their first purchase. For now, we will assume that the customer lifespan is 36 months, or 3 years. Finally, we calculate the basic CLV by multiplying the monthly average revenue and the lifespan. After printing the result, we can see that the average basic CLV is 4774 dollars.

### Granular CLV calculation
Now, we will look into more granular transaction or invoice level data points to calculate the granular customer lifetime value. First, we will calculate average revenue per purchase. We will group on the InvoiceNumber which is a unique purchase, and then calculate the average. As you can see we have called the mean function twice. This is not a mistake. First function call averages the revenue per invoice, and we will have multiple datapoints for each invoice. The second time we call the mean function, we will get a one number that's the overall revenue per purchase average. Next, we calculate the average number of unique invoices per customer each month. We do that by grouping on the CustomerID and the InvoiceMonth, and using the nunique function to count unique number of invoices. Then we add the mean function to average these values to one overall number. Then, we set the lifespan as with the previous example, and then multiply the three values to get the granular CLV, and print it out with some other metrics. We can see that the granular CLV is lower than the basic one at around 1635 dollars. This is a more conservative way to calculate CLV. Let's jump into the traditional CLV calculation where we will get an even smaller CLV estimate.

### Traditional CLV calculation
Alright. Now we will calculate the customer lifetime value with the traditional method which does not require lifespan to be defined, and instead uses retention to churn rate to assess customer life expectancy. We calculate the monthly revenue as we did with the basic CLV. Then we calculate the retention rate from the monthly cohort retention dataset. Here, we exclude the first column, since the retention there is 100% given that in the first month every cohort is 100% active by definition. Then, we calculate the average monthly, and call the mean function the second time to get the overall number. Afterwards, we calculate the churn rate which is just 1 minus retention. Finally, we multiply the average monthly revenue with the retention to churn rate, and get the traditional CLV value. Let's print it out together with the inputs to assess it. We can see that the traditional CLV is significantly lower than the other two measures. The root cause is that previously we used a pre-defined customer lifespan, and here the customer life expectation is inferred from the retention to churn ratio. The retention is very low, therefore the multiplier is less than 1. Typically, retention numbers are higher, somewhere around 80-90% which would roughly make this CLV value between 500 and 1200 dollars respectively. This model assumes that the churn is final, i.e. customers who don't come back the next month, are not coming back in the later periods. We won't explore the retention definition here, but you can test different time periods like quarterly or even annual retention for this and other datasets to assess impact on the retention and churn values.

### Which method to use?
Now, these are just a few models on top of other more statistical approaches that we're not covering in this course. The choice of the formula depends on the business type and the main goal. One thing to assess with the traditional CLV model is that the churn is assumed to be definitive here - i.e. the customer is expected to not come back if they have churned once. This assumption must be validated prior to using this approach. As you've seen in the calculation, the model is not robust at low retention values as the reported customer lifetime values will be too low, even lower than the average monthly revenue spend. Overall, that hardest thing to predict when approaching lifetime value calculation is the frequency of purchases in the future. In the next lesson we will learn how to do that using regression models.

#### Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID','InvoiceMonth'])['TotalSum'].sum().mean()

Calculate average monthly spend
monthly_revenue = np.mean(monthly_revenue)

#### Define lifespan to 36 months
lifespan_months = 36

#### Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months

#### Print the basic CLV value
print('Average basic CLV is {:.1f} USD'.format(clv_basic))

#### Calculate average revenue per invoice
revenue_per_purchase = online.groupby(['InvoiceNo'])['TotalSum'].mean().mean()

#### Calculate average number of unique invoices per customer per month
frequency_per_month = online.groupby(['CustomerID','InvoiceMonth'])['InvoiceNo'].nunique().mean()

#### Define lifespan to 36 months
lifespan_months = 36

#### Calculate granular CLV
clv_granular = revenue_per_purchase * frequency_per_month * lifespan_months

#### Print granular CLV value
print('Average granular CLV is {:.1f} USD'.format(clv_granular))

#### Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID','InvoiceMonth'])['TotalSum'].sum().mean()

#### Calculate average monthly retention rate
retention_rate = retention.iloc[:,1:].mean().mean()

#### Calculate average monthly churn rate
churn_rate = 1 - retention_rate

#### Calculate traditional CLV 
clv_traditional = monthly_revenue * (retention_rate / churn_rate)

#### Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} USD at {:.1f} % retention_rate'.format(clv_traditional, retention_rate*100))
