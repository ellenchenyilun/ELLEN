# Welcome the Elle's Sample Work

Please check out in here one Data-science related sample work of Ellen ;)


## Capital Bike Rental Analysis

The is the analysis for one bike rental company for Washington D.C.


### Import Related Packadge
```markdown
`
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.api import OLS
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import seaborn as sns

bikes_df = pd.read_csv('data/BSS_hour_raw.csv')
print(bikes_df.dtypes)
bikes_df.describe()

bikes_df['dteday'] = pd.to_datetime(bikes_df['dteday'])
print(bikes_df.dtypes)
bikes_df.head()

`
```
![Image image2](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_2.png)


### Explore How Bike Rideship Varies With Hour of The Day

```markdown
`
bikes_df['year'] = bikes_df['dteday'].dt.year
bikes_df['month'] = bikes_df['dteday'].dt.month
bikes_df['dteday'] = bikes_df['dteday'].dt.day
bikes_df['counts'] = bikes_df['casual'] + bikes_df['registered']

def make_violin(df, var1, var2=0):
    if var2==0:
        cur_x_pos = 0
        for p, c in df:
            violin_1 = plt.violinplot(c[var1], 
            positions=[cur_x_pos], 
            showextrema = False, 
            showmedians = False, 
            showmeans = False)
            
            [pc.set_color('#D43F3A') for pc in violin_1['bodies']]
            cur_x_pos+=1
        return violin_1
    
    else: 
        cur_x_pos = 0
        for p, c in df:
            violin_1 = plt.violinplot(c[var1], 
            positions=[cur_x_pos-0.25], 
            showextrema = False, 
            showmedians = False, 
            showmeans = False)
            violin_2 = plt.violinplot(c[var2], 
            positions=[cur_x_pos+0.25], showextrema = False, 
            showmedians = False, 
            showmeans = False)
            
            [pc.set_color('#D43F3A') for pc in violin_1['bodies']]
            [pc.set_color('#2222ff') for pc in violin_2['bodies']]
            
            cur_x_pos+=1
        return violin_1, violin_2

fig, ax = plt.subplots(1,1, figsize=(15,5))
make_violin(bikes_df.groupby('hour'), 'casual', 'registered')
casual = bikes_df.groupby('hour')['casual'].mean()
registered = bikes_df.groupby('hour')['registered'].mean()
ax.plot(casual.index-0.25, casual.values, color = '#D43F3A', label='casual(mean)')
ax.plot(registered.index+0.25, registered.values, color = '#2222ff', label='registered(mean)')
ax.set_xticks(range(24))
ax.set_xlabel('hour of the day', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.legend(fontsize=13, loc = 'upper left')
ax.set_title('Comparing Casual&Registered Rentals During the Day', fontsize = 15)
plt.show()
`
```
![Image image3](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_3.png)

```markdown
`
holiday_df = bikes_df[bikes_df.holiday == 1]
nonholiday_df = bikes_df[bikes_df.holiday == 0]

fig, ax = plt.subplots(1,1, figsize=(15,5))
make_violin(bikes_df.groupby('hour'), 'casual', 'registered')

casual_ho = holiday_df.groupby('hour')['casual'].mean()
registered_ho = holiday_df.groupby('hour')['registered'].mean()
casual_nonho = nonholiday_df.groupby('hour')['casual'].mean()
registered_nonho = nonholiday_df.groupby('hour')['registered'].mean()

ax.plot(casual_ho.index-0.25, casual_ho.values, color = '#D43F3A', alpha=0.8, ls='--', label='Holiday: casual(mean)')
ax.plot(casual_nonho.index-0.25, casual_nonho.values, color = '#D43F3A', label='Non-holiday: casual(mean)')
ax.plot(registered_ho.index+0.25, registered_ho.values, color = '#2222ff', alpha=0.8,ls='--', label='Holiday: registered(mean)')
ax.plot(registered_nonho.index+0.25, registered_nonho.values, color = '#2222ff', label='Non-holiday: registerd(mean)')

ax.set_xticks(range(24))
ax.set_xlabel('hour of the day', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.legend(fontsize=13, loc = 'upper left')
ax.set_title('Comparing Casual&Registered Rentals During the Day for Holiday&Non-holiday', fontsize = 15)
plt.show()
`
```
![Image image4](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_4.png)

```markdown
`
fig, ax = plt.subplots(1,1, figsize=(10,5))
make_violin(bikes_df.groupby('weather'), 'casual', 'registered')
casual = bikes_df.groupby('weather')['casual'].mean()
registered = bikes_df.groupby('weather')['registered'].mean()
ax.scatter((casual.index)-1.25, casual.values, color = '#D43F3A', alpha = 0.8, label='casual(mean by hour)')
ax.scatter((registered.index)-0.75, registered.values, color = '#2222ff', alpha = 0.8, label='registered(mean by hour)')
ax.set_xticks(range(4))
ax.set_xticklabels(['Sunny', 'Cloudy', 'Snow', 'Storm'])
ax.set_xlabel('weather type', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.legend(fontsize=13)
ax.set_title('Comparing Casual&Registered Rentals in Different Weather', fontsize = 15)
plt.show()
`
```

![Image image5](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_5.png)


### Explore Seasonality on Bike Ridership

```markdown
`
df = pd.read_csv('data/BSS_hour_raw.csv')
bikes_by_day = df.groupby('dteday').agg({'weekday': 'mean', 'weather': 'max', 'season': 'mean', 'temp': 'mean', 'atemp':'mean', 'windspeed': 'mean', 'hum':'mean', 'casual': 'sum', 'registered':'sum' })
bikes_by_day['counts'] = bikes_by_day.casual + bikes_by_day.registered
bikes_by_day = bikes_by_day.reset_index()

fig, ax = plt.subplots(1,1, figsize=(10,5))
make_violin(bikes_by_day.groupby('season'), 'casual', 'registered')
casual = bikes_by_day.groupby('season')['casual'].mean()
registered = bikes_by_day.groupby('season')['registered'].mean()
ax.plot((casual.index)-1.25, casual.values, color = '#D43F3A', alpha = 0.8, marker = 'o', label='casual(Avg by day)')
ax.plot((registered.index)-0.75, registered.values, color = '#2222ff', alpha = 0.8, marker = 'o', label='registered(Avg by day)')
ax.set_xticks(range(4))
ax.set_xticklabels( ['winter', 'spring', 'summer', 'fall'])
ax.set_xlabel('season', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.legend(fontsize=13, loc='upper left')
ax.set_title('Comparing Casual&Registered Rentals in Different Seasons', fontsize = 15)
plt.show()
`
```
![Image image7](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_7.png)

```markdown
`
bikes_by_day['registered_to_counts'] = bikes_by_day['registered']/bikes_by_day['counts']
fig, ax = plt.subplots(1,1, figsize=(10,5))
make_violin(bikes_by_day.groupby('weekday'), 'registered_to_counts')
regtoc = bikes_by_day.groupby('weekday')['registered_to_counts'].mean()
ax.plot((regtoc.index), regtoc.values, color = '#D43F3A', alpha = 0.8, marker = 'o')
ax.set_xticks(range(7))
ax.set_xticklabels(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
ax.set_xlabel('weekday', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.set_title('Registered Rental Percentatge in Different Weekdays', fontsize = 15)
plt.show()
`
```
![Image image8](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_8.png)

```markdown
`
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.hist((bikes_by_day.groupby('weather').get_group(2))['counts'], alpha = 0.3, color = 'black', label = 'cloudy')
ax.hist((bikes_by_day.groupby('weather').get_group(1))['counts'], alpha = 0.3, color = 'red', label = 'sunny')
ax.set_xlabel('number of total rentals', fontsize = 13)
ax.set_ylabel('frequency', fontsize = 13)
ax.set_title('Histogram of Total Rentals in Different Weathers', fontsize = 15)
ax.legend(fontsize = 13)
plt.show()
`
```
![Image image9](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_9.png)

```markdown
`fig, ax = plt.subplots(1,1, figsize=(8,5))
cur_x_pos = 0
for p , c in bikes_by_day.groupby('season'):
    box = plt.boxplot(c['counts'], positions=[cur_x_pos], flierprops={'alpha':0.6,'markersize': 10, 'markerfacecolor':'red', 'markeredgecolor': 'None'})
    cur_x_pos+=1
ax.set_xticks(range(4))
ax.set_xticklabels(['winter', 'spring', 'summer', 'fall'])
ax.set_xlabel('season', fontsize = 13)
ax.set_ylabel('number of rentals', fontsize = 13)
ax.set_title('Distribution of Rentals in Different Seasons', fontsize = 15)
plt.xlim(-1,4)
plt.show()
`
```
![Image image10](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_10.png)

### Prepare the data for Regression
```markdown
`col = ['weekday', 'season', 'month', 'weather', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'counts']
scatter_matrix(bikes_df[col], figsize = (20,10))
plt.show()

corr = bikes_df[col].corr()
corr.style.background_gradient()
`
```
![Image image11](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_11.png)

```markdown
`col_cat = ['weekday', 'season', 'month', 'weather']
bikes_df_onehot = pd.get_dummies(bikes_df, columns=col_cat, drop_first=True)
train_data, test_data = train_test_split(bikes_df, test_size = 0.2, stratify=bikes_df['month'])

BSS_train = pd.read_csv('data/BSS_train.csv')
BSS_test = pd.read_csv('data/BSS_test.csv')
col_drop = ['dteday', 'Unnamed: 0', 'year', 'registered', 'casual']
BSS_train = BSS_train.drop(col_drop, axis=1)
BSS_test = BSS_test.drop(col_drop, axis=1)

corr_train = BSS_train.corr()
corr_train.style.background_gradient()

rental_pos = corr_train['counts'][corr_train['counts'].values > 0].index
print(rental_pos)

stack = corr_train.stack()
big_cor = [stack.index[x] for x in np.where(stack.values>0.7)[0]]
big_cor_real = [(x,y) for x, y in big_cor if x!=y]
big_cor_real
`
```
### Multiple Linear Regression

```markdown
`
X_train = BSS_train.drop('counts', axis = 1)
X_train = sm.add_constant(X_train)
y_train = BSS_train.counts
X_test = BSS_test.drop('counts', axis = 1)
X_test = sm.add_constant(X_test)
y_test = BSS_test.counts

BSS_sm_ols = OLS(y_train, X_train).fit()
train_r2 = BSS_sm_ols.rsquared

BSS_sm_preditions_test = BSS_sm_ols.predict(X_test)
test_r2 = r2_score(y_test, BSS_sm_preditions_test)

BSS_sm_ols.pvalues[BSS_sm_ols.pvalues<0.05].index

res = BSS_sm_ols.resid
fitted_values = BSS_sm_ols.fittedvalues
fig, ax = plt.subplots(figsize = (15, 5))
ax.scatter(fitted_values, res, s=1, label='residuals')
ax.plot([min(fitted_values), max(fitted_values)],[0,0], c='red', lw=1.2, label='zero line')
ax.set_xlabel('fitted values', fontsize=13)
ax.set_ylabel('residuals', fontsize=13)
ax.set_title('Residual Plot', fontsize=15)
ax.legend(fontsize=13, loc='upper left')
plt.show()
`
```
![Image image12](https://github.com/ellenchenyilun/Ellen/raw/master/image/image_12.png)


### Model Selection
```markdown
`
def get_bic(X_train, y_train):
    model = OLS(y_train, X_train).fit()
    return model.bic

variables_exclude = [x for x in X_train.columns.values if x!='const']
best_bic = []
variables_include = []
best_variables = []

for n in range(30):
    bic_set = []
    variables_set = []
    for v in variables_exclude:
        col = variables_include + [v]
        bic = get_bic(sm.add_constant(X_train[col]), y_train)
        bic_set.append(bic)
        variables_set.append(v)

    best_bic.append(np.min(bic_set))
    best_variable = variables_set[bic_set.index(np.min(bic_set))]
    variables_include += [best_variable]
    variables_exclude = [x for x in variables_exclude if x != best_variable]
    best_variables.append(set(variables_include))

    
best_model = sorted(zip(best_variables, best_bic), key=lambda x: x[1])[0]
forward_predicors = list(best_model[0])
print(forward_predicors)

col = ['Jul', 'Jun', 'hour', 'Snow', 'holiday', 'Cloudy', 'hum', 'temp', 'Aug', 'fall']
BSS_sm_ols_new = OLS(y_train, sm.add_constant(X_train[col])).fit()
train_r2_new = BSS_sm_ols_new.rsquared

BSS_sm_preditions_test_new = BSS_sm_ols_new.predict(sm.add_constant(X_test[col]))
test_r2_new = r2_score(y_test, BSS_sm_preditions_test_new)

continuous_predictors = [['hour'], ['temp'], ['atemp'], ['hum'], ['windspeed']]
X_train_polynomial = []
for n in continuous_predictors:
    X_train_poly = X_train[n]
    transformer_4 = PolynomialFeatures(4, include_bias=False)
    new_features = transformer_4.fit_transform(X_train_poly)
    X_train_polynomial.append(new_features)

X_train_poly = np.concatenate((X_train_polynomial[0],X_train_polynomial[1], X_train_polynomial[2], X_train_polynomial[3],X_train_polynomial[4]), axis=1)

model_poly = OLS(y_train, sm.add_constant(X_train_poly)).fit()
r2_train_poly = model_poly.rsquared

X_test_polynomial = []
for n in continuous_predictors:
    X_test_poly = X_test[n]
    transformer_4 = PolynomialFeatures(4, include_bias=False)
    new_features = transformer_4.fit_transform(X_test_poly)
    X_test_polynomial.append(new_features)

X_test_poly = np.concatenate((X_test_polynomial[0],X_test_polynomial[1], X_test_polynomial[2], X_test_polynomial[3],X_test_polynomial[4]), axis=1)


preditions_test_poly = model_poly.predict(sm.add_constant(X_test_poly))
r2_test_poly = r2_score(y_test, preditions_test_poly)

print("For polynomial model: ['hour', 'hour^2', 'hour^3', 'hour^4','temp', 'temp^2', 'temp^3', 'temp^4', 'atemp', 'atemp^2', 'atemp^3', 'atemp^4', 'hum', 'hum^2', 'hum^3', 'hum^4', 'windspeed', 'windspeed^2', 'windspeed^3', 'windspeed^4']")

model_poly.pvalues[model_poly.pvalues<0.05].index
`
```

### Report to Company

From the previous analyis, there are some predictors that are significant and effecitive in predicting the total rentals, such as "hour", "holiday", "temp", "weather" and so on. Frome the scatter matrix, it is not difficult to see that most of the predictors has no or ambiguous linear relations with the total rentals, except the "casual" and "registered" predictors, which actually are what compute the total rentals. Hence, adopting a linear model is hesitate. Additionaly, after running the multiple linear regression model, the result is not good. The test R2 of both full model and best model from forward selection equals to around 0.35, which is not desiarable. Therefore, an non-linear model is suggested. When selecting 5 continuous predictor: 'hour', 'temp', 'atemp', 'hum', 'windspeed' with their 1th ,2th ,3th and 4th power combined for a polynimial model, we get a higher test R2 at 0.47, which is better than that of the multiple linear regression. However, in this model, we ignore some seemingly highly related catigorical predictors, such as: holiday, weather, and month. Hence, this model can actually be further modified.

First of all, it is important to ensure the two peak hour supply especially for registered riders, they are of high chance to be commuer since a significant percent of ridership are driven from these two peak hours. Further reseaches such as riders demographic information and location can be conducted to address this problem. Secondly, the company can provide some special and innovative "bad weather day equipment" in the bysical station or even in the bike for people to go over the "bad weather" since from the analysis, the ridership drcrease in snow and strom weathers. Thirdly, winter promotion may worth a try, for example, sponsoring winter "cycling competion" or provide encentives for people to do cycling in winter day. This strategy will help to increase winter ridership, which is lower than that of other seasons. Forthly, the company should enhance the loyalty programme to keep the registered riders in hand since they are a large percent of the total riders.

### Reference
1. [Capital Bikeshare program](https://www.capitalbikeshare.com)
