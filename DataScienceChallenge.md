
# Shopify Fall 2022 Data Science Intern Challenge

*Alan Milligan - UBC BSc Combined Honours Computer Science and Mathematics*

##  Question 1

### Findings

#### a)
Average Order Value (AOV) as described by the [Shopify blog](https://www.shopify.ca/blog/average-order-value) can be written as follows

$$
\frac{1}{n}\sum_{i=1}^N(\text{value of order } i) = \frac{1}{n}\sum_{i=1}^N\left(\sum_{j=1}^{k_i}(\text{unit cost } j)\cdot(\text{quantity } j)\right)
$$

Taking a look at this, I see two major ways in which is calculation might become misleading. If any order happens have many items (having $k_i$ or quantity $j$ very large in the math) then it can  influence the average significantly or if any order contains a very expensive item (having unit cost $j$ very large) then it too could throw off the AOV. A combination of these issues would make things even worse. Both of these issues stem from the fact that the mean is very sensitive to outliers (the same issue that makes least squares regression fail with outliers).

In this dataset, both of the previously mentioned issues occur, leading to the AOV being very misleading for the average sneaker shop. I will spotlight two shops who appear to be responsible for these issues.

**Shop 42**

Throughout March of 2017, the shop with ID 42 submits 17 orders having order amount \\$704000 and total items 2000. This would likley be some sort of bulk order of similar items, each having an average value of \\$352. While this average could suffer the same  ambiguity described above, it is likley that we are in the first failure case, where the sheer volume of an order is influencing the AOV. In fact, this shop has several orders for 1 item with the order amount being \\$352 (and other small orders with the same average), supporting the hypothesis they made  a bulk order of one medium cost shoe.

**Shop 78**

On the other side of spectrum, all of 78's orders are below 6 items (the majority being 1 or 2 items) each order having average unit cost \\$25725. This very likley means we have a shop selling only the fanciest of fancy shoes and so their order amounts are significantly throwing off the AOV via the second described failure case.

We can look at the summary statistics of order amount with all shops and excluding the above two shops here.


|Summary Statistic   |    All Data| Excluding Shop 42 and 78 |
|  :-- | --- |  --- |
|Count               | 5000| 4903.00     | 
|AOV                 |   3145.13| 300.16 |
|Standard Deviation  |  41282.54| 155.94 |  
|Minimum             |    90.00| 90.00   |
|25% Quartile        |    163.00| 163.00 |  
|50% Quartile        |    284.00| 284.00 | 
|75% Quartile        |    490.00| 386.50 |  
|Maximum             | 704000.00| 1086.00|

While we can see the AOV is now roughly \\$300, which seems much more realistic for sneaker orders, another statistic to note is the standard deviation (which is the expected difference from the AOV) is has dropped from over \\$40000 to about \\$155. This suggest that the new AOV is closer to much more of the remaining  data. These ideas will  inform the next two answers, and were further examined in the analysis section.

#### b)
The main problem we found with AOV was that it is sensitive to outliers, and in this there are outliers that  are inflating it. In situations where this is the case we often use a metric called the *trimmed mean*. Classically we would remove some upper and lower quantile when dealing with values coming from a symmetric distribiton. Since we are dealing with  costs that are stricly positive, and the lower quartile is already relativly close to zero,  we can  use a one sided trimmed mean and remove some upper quantile of the data to create *trimmed Average Order Value*. This is of course under the asumption I can only report one metric. As mentioned in the blog above, a single statistic, no matter what it is, can always be problematic. I would prefer to report several summary statistics such as the table above, but if we need to boil it down to one metric then we can try and make a form of AOV that is as reprentative as possible.

By examining the summary statistics above, specifically the quartile ranges, we can assume that the outliers we are dealing are very few but very large, so trimming a small amount should allow us to effectivly capture the majority  of the "normal" orders. This would be ideal from a business perspective as whatever analysis that applies to these normal orders is likley not useful for shops 42 and 78, who appear to be operating a different sort of business than the other shops. We will chose to keep the bottom 95\% of the data sorted by order amount.

#### c)

The 0.95\%-trimmed AOV is \\$284.7. Interestingly, this (up to rounding) is the same as the median order value of full dataset, which I consider a good sign that this is a reasonable estimate of what the average sneaker order would look like. We also see that the range of the data is now \\$550, as opposed to over \\$700000 previously, which seems more reasonable in this context. With this adjusted metric we have a better idea of how much a customer spends on a given order. Below are the other statistics I would report if I could use more than one metric.

|Summary Statistic   |   95\%-trimmed Data|
|  :-- | --- |
|Count               | 4750|
|AOV                 |   284.70|
|Standard Deviation  |  133.08|
|Minimum             |    90.00|
|25\% Quartile        |    161.00|
|50\% Quartile        |    272.00|
|75\% Quartile        |    362.00|
|Maximum             | 640.00|



### Analysis

My first step with any data analysis or machine learning task is to simply take a look at the data. I can see what sort of data we are dealing with, if there are any immediate red flags, and decide what to look at  next.


```python
import pandas as pd
import matplotlib.pyplot as plt

dataset_name = "2019 Winter Data Science Intern Challenge Data Set - Sheet1.csv"
data = pd.read_csv(dataset_name)
data.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>746</td>
      <td>224</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-13 12:36:56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>92</td>
      <td>925</td>
      <td>90</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-03 17:38:52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>44</td>
      <td>861</td>
      <td>144</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-14 4:23:56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>935</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-26 12:43:37</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18</td>
      <td>883</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-01 4:35:11</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>58</td>
      <td>882</td>
      <td>138</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-14 15:25:01</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>87</td>
      <td>915</td>
      <td>149</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-01 21:37:57</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>22</td>
      <td>761</td>
      <td>292</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-08 2:05:38</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>64</td>
      <td>914</td>
      <td>266</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-17 20:56:50</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>52</td>
      <td>788</td>
      <td>146</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-30 21:08:26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>66</td>
      <td>848</td>
      <td>322</td>
      <td>2</td>
      <td>credit_card</td>
      <td>2017-03-26 23:36:40</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>40</td>
      <td>983</td>
      <td>322</td>
      <td>2</td>
      <td>debit</td>
      <td>2017-03-12 17:58:30</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>54</td>
      <td>799</td>
      <td>266</td>
      <td>2</td>
      <td>credit_card</td>
      <td>2017-03-16 14:15:34</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>100</td>
      <td>709</td>
      <td>111</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-22 2:39:49</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>87</td>
      <td>849</td>
      <td>447</td>
      <td>3</td>
      <td>credit_card</td>
      <td>2017-03-10 11:23:18</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-07 4:00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>17</td>
      <td>731</td>
      <td>176</td>
      <td>1</td>
      <td>cash</td>
      <td>2017-03-21 4:23:38</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>28</td>
      <td>752</td>
      <td>164</td>
      <td>1</td>
      <td>credit_card</td>
      <td>2017-03-21 12:09:07</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>83</td>
      <td>761</td>
      <td>258</td>
      <td>2</td>
      <td>cash</td>
      <td>2017-03-17 13:18:47</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>63</td>
      <td>898</td>
      <td>408</td>
      <td>3</td>
      <td>credit_card</td>
      <td>2017-03-29 15:11:52</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[["order_amount","total_items"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_amount</th>
      <th>total_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3145.128000</td>
      <td>8.78720</td>
    </tr>
    <tr>
      <th>std</th>
      <td>41282.539349</td>
      <td>116.32032</td>
    </tr>
    <tr>
      <th>min</th>
      <td>90.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>163.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>284.000000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>390.000000</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>704000.000000</td>
      <td>2000.00000</td>
    </tr>
  </tbody>
</table>
</div>



So it lookslike we have some fairly standard data about orders, but luckily I happened to look at enough to notice one strange entry on line 15. It is clearly an outlier in both the order amount and total items column, which may be a clue to why the origional AOV is so high. Glancing at the summary statistics for our numerical values, we see the origional AOV reported (\\$3145.13), and below it a standard deviation over \\$40000. This is a big red flag to me as that suggests there are orders that are much much larger than the reported AOV, espcially since this is a dollar amount bounded below by 0. The total items standard deviation also seems a bit high, but not quite as extreme. To further examine this hypothesis we can do some visualization.


```python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(data.order_amount)
plt.title("Order Amount")
plt.subplot(1,2,2)
plt.title("Total Items")
plt.hist(data.total_items)
plt.show()
```


![png](output_9_0.png)


So we can see there are some outliers in the data, but the differences in scale make it hard to see what's going on. We can try and see a bit more by using logarithmic scale and using some more bins.


```python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(data.order_amount, log=True, bins=25)
plt.title("Order Amount Frequency")
plt.subplot(1,2,2)
plt.title("Total Items Frequency")
plt.hist(data.total_items, log=True, bins=25)
plt.show()
```


![png](output_11_0.png)


So here we can see whats going on a little better. There is very clearly some outliers in order amount at round \\$700000, and likewise in total items around 2000. We can also see there is somewhat of an outlier in order amount near \\$150000, which is again quite a bit for a shoe order. From here I want to see if there is anything interesting about these outlier datapoints, which we can do by looking at the sorted data. By looking at the histogram, I can see there is around 15 datapoints with the around \\$700000 value, so I will try and look a bit past that.


```python
data.sort_values(by=["order_amount"], ascending=False).head(30)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2153</th>
      <td>2154</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-12 4:00:00</td>
    </tr>
    <tr>
      <th>3332</th>
      <td>3333</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-24 4:00:00</td>
    </tr>
    <tr>
      <th>520</th>
      <td>521</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-02 4:00:00</td>
    </tr>
    <tr>
      <th>1602</th>
      <td>1603</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-17 4:00:00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>61</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-04 4:00:00</td>
    </tr>
    <tr>
      <th>2835</th>
      <td>2836</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-28 4:00:00</td>
    </tr>
    <tr>
      <th>4646</th>
      <td>4647</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-02 4:00:00</td>
    </tr>
    <tr>
      <th>2297</th>
      <td>2298</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-07 4:00:00</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>1437</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-11 4:00:00</td>
    </tr>
    <tr>
      <th>4882</th>
      <td>4883</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-25 4:00:00</td>
    </tr>
    <tr>
      <th>4056</th>
      <td>4057</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-28 4:00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-07 4:00:00</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>1105</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-24 4:00:00</td>
    </tr>
    <tr>
      <th>1562</th>
      <td>1563</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-19 4:00:00</td>
    </tr>
    <tr>
      <th>2969</th>
      <td>2970</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-28 4:00:00</td>
    </tr>
    <tr>
      <th>4868</th>
      <td>4869</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-22 4:00:00</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>1363</td>
      <td>42</td>
      <td>607</td>
      <td>704000</td>
      <td>2000</td>
      <td>credit_card</td>
      <td>2017-03-15 4:00:00</td>
    </tr>
    <tr>
      <th>691</th>
      <td>692</td>
      <td>78</td>
      <td>878</td>
      <td>154350</td>
      <td>6</td>
      <td>debit</td>
      <td>2017-03-27 22:51:43</td>
    </tr>
    <tr>
      <th>2492</th>
      <td>2493</td>
      <td>78</td>
      <td>834</td>
      <td>102900</td>
      <td>4</td>
      <td>debit</td>
      <td>2017-03-04 4:37:34</td>
    </tr>
    <tr>
      <th>3724</th>
      <td>3725</td>
      <td>78</td>
      <td>766</td>
      <td>77175</td>
      <td>3</td>
      <td>credit_card</td>
      <td>2017-03-16 14:13:26</td>
    </tr>
    <tr>
      <th>4420</th>
      <td>4421</td>
      <td>78</td>
      <td>969</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-09 15:21:35</td>
    </tr>
    <tr>
      <th>4192</th>
      <td>4193</td>
      <td>78</td>
      <td>787</td>
      <td>77175</td>
      <td>3</td>
      <td>credit_card</td>
      <td>2017-03-18 9:25:32</td>
    </tr>
    <tr>
      <th>3403</th>
      <td>3404</td>
      <td>78</td>
      <td>928</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-16 9:45:05</td>
    </tr>
    <tr>
      <th>2690</th>
      <td>2691</td>
      <td>78</td>
      <td>962</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-22 7:33:25</td>
    </tr>
    <tr>
      <th>2564</th>
      <td>2565</td>
      <td>78</td>
      <td>915</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-25 1:19:35</td>
    </tr>
    <tr>
      <th>4715</th>
      <td>4716</td>
      <td>78</td>
      <td>818</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-05 5:10:44</td>
    </tr>
    <tr>
      <th>1259</th>
      <td>1260</td>
      <td>78</td>
      <td>775</td>
      <td>77175</td>
      <td>3</td>
      <td>credit_card</td>
      <td>2017-03-27 9:27:20</td>
    </tr>
    <tr>
      <th>2906</th>
      <td>2907</td>
      <td>78</td>
      <td>817</td>
      <td>77175</td>
      <td>3</td>
      <td>debit</td>
      <td>2017-03-16 3:45:46</td>
    </tr>
    <tr>
      <th>3705</th>
      <td>3706</td>
      <td>78</td>
      <td>828</td>
      <td>51450</td>
      <td>2</td>
      <td>credit_card</td>
      <td>2017-03-14 20:43:15</td>
    </tr>
    <tr>
      <th>3101</th>
      <td>3102</td>
      <td>78</td>
      <td>855</td>
      <td>51450</td>
      <td>2</td>
      <td>credit_card</td>
      <td>2017-03-21 5:10:34</td>
    </tr>
  </tbody>
</table>
</div>



Great, so we can see all of the \\$700000 order amount outliers, and notice that they are also (at least part of) the outliers in order amount. They are also coming from the same shop, so it seems this is some sort of bulk order as opposed to just selling to average consumers. When we looked at the data earlier it appeared that most of the orders were small and likley headed to customers, so it may make sense to exclude shop 78 if it is a different type of shoe business. This got me thinking about AOV per store, rather than across all stores. We can calculate that and take a look as follows.


```python
data.groupby(["shop_id"])["order_amount"].mean().sort_values()
```




    shop_id
    92        162.857143
    2         174.327273
    32        189.976190
    100       213.675000
    53        214.117647
               ...      
    38        390.857143
    90        403.224490
    50        403.545455
    78      49213.043478
    42     235101.490196
    Name: order_amount, Length: 100, dtype: float64



Now things are starting to make sense. We have two shops (78 and 42) with huge AOV, where all the other shops are between \\$405 and \\$160, which seem like reasonable average order values for a sneaker shop. This makes me wonder what the data would look like if we excluded these two shops, which we can do.


```python
filtered_data =  data[(data.shop_id != 78) &  (data.shop_id != 42)]
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(filtered_data.order_amount, log=False, bins=25)
plt.title("Order Amount Frequency")
plt.subplot(1,2,2)
plt.title("Total Items Frequency")
plt.hist(filtered_data.total_items, log=False, bins=25)
plt.show()
```


![png](output_17_0.png)


Now this looks a bit more normal, roughly following the rank-frequency inverse relationship often seen in data. That being said, I don't feel like the best rule is just kick out two shops after looking at the data. If we wanted to expand this dataset we might quickly run into the same issue if another outlier shop shows up. Instead, we can take a slightly more principled approach and use something like the trimmed mean, often used in situations where we would like an average of sorts but don't want to a victim of outliers. Often this is done by something like an interquartile mean where we only take the mean in the interquartile range, but looking at this data I think only trimming the upper end is appropriate. This is because we know that both order amount and total items are bounded below by zero, so we don't have to worry about running into a massive negative value. I will take the values below the 95\% quantile given that it seems nearly all of our data is "normal" orders rather than outliers.


```python
total_records = data.count()[0]
kept_records = int(0.95*total_records)
trimmed_data = data.sort_values(["order_amount"])[0:kept_records]
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.hist(trimmed_data.order_amount, log=False, bins=25)
plt.title("Order Amount Frequency")
plt.subplot(1,2,2)
plt.title("Total Items Frequency")
plt.hist(trimmed_data.total_items, log=False, bins=25)
plt.show()
```


![png](output_19_0.png)


This looks more like something we can make business decisions with, and we can see from the summary statistics below the standard deviation is now a level that makes sense in this context (I don't know much about shoes but \$133 seems like it could be difference between 2 pairs and 3 pairs).


```python
trimmed_data[["order_amount","total_items"]].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_amount</th>
      <th>total_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4750.000000</td>
      <td>4750.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>284.705684</td>
      <td>1.903789</td>
    </tr>
    <tr>
      <th>std</th>
      <td>133.079491</td>
      <td>0.868124</td>
    </tr>
    <tr>
      <th>min</th>
      <td>90.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>161.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>272.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>362.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>640.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Question 2

For the following queries I am going to use "manual" joins on tables' primary keys rather than SQL's built in JOIN keyword and its variants as in the case of the following queries it is a bit more succinct. The result of each query is displayed as a table following the query.

#### a)


```SQL
SELECT COUNT(O.OrderID) AS SpeedyExpressOrders
FROM Orders O, Shippers S
WHERE O.ShipperID = S.ShipperID
AND S.ShipperName = "Speedy Express";
```

<table class="ws-table-all notranslate"><tbody><tr><th>SpeedyExpressOrders</th></tr><tr><td>54</td></tr></tbody></table>

Here we join the Orders table and Shippers table by ShipperID so that each record (corresponding to an order) will have its shipper information attached. Then, we keep only the records that have ShipperName "Speedy Express." Finally, we count the number of orders remaining by their unique primary key OrderID to reach the final answer.

#### b)

```SQL
SELECT E.LastName, COUNT(O.OrderID) AS OrderCount 
FROM Employees E, Orders O
WHERE E.EmployeeID = O.EmployeeID
GROUP BY E.EmployeeID 
ORDER BY OrderCount DESC
LIMIT 1;
```


<table class="ws-table-all notranslate"><tbody><tr><th>LastName</th><th>OrderCount</th></tr><tr><td>Peacock</td><td>40</td></tr></tbody></table>

Similar with the previous query, we manually join the Employees and Orders tables on Employee ID so that each order will have the name of the employee handling it attached. From this, we group on each employee and aggregate the number of orders handled by this employee into the new column OrderCount. We then order the remaining records by OrderCount from largest to smallest, and keep the top record as our result.

#### c)

```SQL
SELECT P.ProductName, SUM(OD.Quantity) AS ProductsOrdered 
FROM Orders O, OrderDetails OD, Customers C, Products P
WHERE C.CustomerID = O.CustomerID
AND O.OrderID = OD.OrderID
AND OD.ProductID = P.ProductID
AND C.Country = "Germany"
GROUP BY P.ProductID
ORDER BY SUM(OD.Quantity) DESC
LIMIT 1;
```

<table class="ws-table-all notranslate"><tbody><tr><th>ProductName</th><th>ProductsOrdered</th></tr><tr><td>Boston Crab Meat</td><td>160</td></tr></tbody></table>

*Assumption: "ordered the most by customers in Germany" means of all products, this product had the most **units** sent to Germany, **not** the most **orders** originating from Germany*

We start by doing a very large join between Orders, OrderDetails, Customers, and Products in order to connect Country (in the Customers table) to ProductID (in the Products table) and get access to Quantity (in the OrderDetails table) along the way. Then, we filter to only have records of orders from customers in Germany as desired, and we group by ProductID aggregating by adding up all the units sold for each product into a new column called ProductsOrdered. We then sort by the ProductsOrdered column and take the ProductName and  ProductsOrdered from the top entry as our answer.
