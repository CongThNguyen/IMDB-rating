# Machine Learning in Python - Project 1

Due Friday, March 12th by 5 pm (extension resquest approved).
*contributors names:*

*DUNSTAN SASHA*/ UUN: s1802092

*MAK PO WUN*/ UUN: s2081028

*KURIHARA MASAKI* / UUN: s2121881

*NGUYEN CONG* / UUN: s2133124


## 0. Setup


```python
# Install required packages
!pip install -q -r requirements.txt
```


```python
# Add any additional libraries or submodules below

# Display plots inline
%matplotlib inline

# Data libraries
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn
```


```python
# Load data
d = pd.read_csv("the_office.csv")
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
      <th>season</th>
      <th>episode</th>
      <th>episode_name</th>
      <th>director</th>
      <th>writer</th>
      <th>imdb_rating</th>
      <th>total_votes</th>
      <th>air_date</th>
      <th>n_lines</th>
      <th>n_directions</th>
      <th>n_words</th>
      <th>n_speak_char</th>
      <th>main_chars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Pilot</td>
      <td>Ken Kwapis</td>
      <td>Ricky Gervais;Stephen Merchant;Greg Daniels</td>
      <td>7.6</td>
      <td>3706</td>
      <td>2005-03-24</td>
      <td>229</td>
      <td>27</td>
      <td>2757</td>
      <td>15</td>
      <td>Angela;Dwight;Jim;Kevin;Michael;Oscar;Pam;Phyl...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>Diversity Day</td>
      <td>Ken Kwapis</td>
      <td>B.J. Novak</td>
      <td>8.3</td>
      <td>3566</td>
      <td>2005-03-29</td>
      <td>203</td>
      <td>20</td>
      <td>2808</td>
      <td>12</td>
      <td>Angela;Dwight;Jim;Kelly;Kevin;Michael;Oscar;Pa...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Health Care</td>
      <td>Ken Whittingham</td>
      <td>Paul Lieberstein</td>
      <td>7.9</td>
      <td>2983</td>
      <td>2005-04-05</td>
      <td>244</td>
      <td>21</td>
      <td>2769</td>
      <td>13</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>The Alliance</td>
      <td>Bryan Gordon</td>
      <td>Michael Schur</td>
      <td>8.1</td>
      <td>2886</td>
      <td>2005-04-12</td>
      <td>243</td>
      <td>24</td>
      <td>2939</td>
      <td>14</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>Basketball</td>
      <td>Greg Daniels</td>
      <td>Greg Daniels</td>
      <td>8.4</td>
      <td>3179</td>
      <td>2005-04-19</td>
      <td>230</td>
      <td>49</td>
      <td>2437</td>
      <td>18</td>
      <td>Angela;Darryl;Dwight;Jim;Kevin;Michael;Oscar;P...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>6</td>
      <td>Hot Girl</td>
      <td>Amy Heckerling</td>
      <td>Mindy Kaling</td>
      <td>7.8</td>
      <td>2852</td>
      <td>2005-04-26</td>
      <td>346</td>
      <td>39</td>
      <td>3028</td>
      <td>13</td>
      <td>Angela;Dwight;Jim;Kevin;Michael;Oscar;Pam;Ryan...</td>
    </tr>
  </tbody>
</table>
</div>



## 1. Introduction


NBC Universal wants to create a special reunion episode for the show called "The Office" and they aim to have it rated as high as possible on the IMDb website. For that to happen, they would like to understand the reasons behind the popularity of a few episodes (based on imdb rating) compare to the others.  

Our team will utilize data provided by the studio to develop a model which can predict `imdb_ratings` based on the different number of features (or factors which influence the rating) such as 'directors', 'total votes', etc. In total, there are 13 different features which may or may not contribute to how the show is rated. Therefore, we perform a quick correlations test. The test allows us to decide whether to keep or discard a few features with low correlations concerning to the `imdb_ratings`.

The kept features will be transformed via feature engineering using dummy coding. Dummy coding transforms our independent variables (x) into dummy variables which take a boolean value of either 1 or 0 depending on whether that variable is present or not in a certain episode or seasons. 

Feature transformation undoubtedly increases the total number of features. To counter this, we will combine some features into a group based on the frequencies in which they appear. Features with relatively high frequency will be divided into groups with different conditions. On the other hand, features with frequency levels below a certain threshold will be discarded. With this, we managed to reduce the total number of features down to a manageable size. 

The next step is to fit a regression model onto the transformed data. Having tested the data onto different types of models and comparing the root mean square error(RMSE), our team decided to prioritize Lasso regression. 

Finally, in order to optimize the Lasso model, our team will perform a grid-search method to find the tuning parameter alpha that can minimize the RMSE.




## 2. Exploratory Data Analysis and Feature Engineering

### Preprocessing


Using the schrutepy package, we derive a new feature called `char_most_lines` which represents the character with most lines in each episode.


```python
from schrutepy import schrutepy
df = schrutepy.load_schrute()

seas_eps = list(d.groupby(['season', 'episode']).groups.keys())
char_most_lines=[]
for (s,e) in seas_eps:
    se = df[(df['season'].values == s) & (df['episode'].values == e)].groupby('character').count()[['index']]
   
    char_most_lines.append(se['index'].idxmax())

d['char_most_lines']=char_most_lines
```

To simplify `air_date`, we group the dates according to months and rename the column as `air_month`.


```python
#Replacing 'air_date' with 'air_month'. The months June, July and August are excluded since no episodes are aired during those months
month_list=[('-01-','Jan'),('-02-','Feb'),('-03-', 'Mar'),('-04-','Apr'),('-05-', 'May'),('-09-', 'Sep'),('-10-','Oct'),('-11-','Nov'),('-12-','Dec')]

for i,j in month_list:
    
    ind=d.index[d['air_date'].str.contains(i)]

    for k in d.loc[ind]['air_date']:
        d['air_date']=d['air_date'].replace(k,j)
d=d.rename({'air_date':'air_month'}, axis=1)

```

### Exploratory Data Analysis


```python
# show a pair plot of origina data
sns.pairplot(d)
```




    <seaborn.axisgrid.PairGrid at 0x7ffa8f31d590>




    
![png](project1_files/project1_14_1.png)
    


The scatter plot for `imdb_rating` vs `total_votes` seems to suggest a non-linear relationship while the scatter plot for `n_words` vs `n_lines` indicates that there is a linear relationship with positive slope. The `season` variable appears to be categorical. `total_votes` seems to decrease with `season` while `imdb_rating` appears to fluctuate for each season. There is no clear pattern for the scatter plots of `imdb_rating` vs `n_lines`/`n_directions`/`n_words`/`n_speak_char`. The distributions for the numeric variables are quite skewed, we can therefore apply a log transformation to these variables.


```python
#Applying log transformation due the fact that some distribution is skewed. 
d['total_votes']=np.log(d['total_votes'])
d['n_lines']=np.log(d['n_lines'])
d['n_directions']=np.log(d['n_directions'])
d['n_words']=np.log(d['n_words'])
d['n_speak_char']=np.log(d['n_speak_char'])

```

We now look at the correlation between variables by plotting the correlation heat map below:


```python
#Perform a correlation test on the data and present it in the form of a correlation matrix.
corrmat = d.corr()
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)
```




    <AxesSubplot:>




    
![png](project1_files/project1_18_1.png)
    





Based on the plot above, we observe that the correlation between `total_votes` and `imdb_rating` is quite high. The correlation between `imdb_ratimg` and the other variables seem to be low. We employ the filter method for feature selection, whereby variables whose correlation value(taken to be the absolute value) are lower than a certain threshold is dropped. In this case, we set the threshold to be 0.20, resulting in the variables `episode`, `n_directions` and `n_speak_char` to be dropped.


```python
#This code allow us to "drop" the features which show little or no correlation to the outcomes 
d=d.drop(['episode', 'n_directions', 'n_speak_char'], axis=1)
```



Looking at the correlation between predictor variables, there is a rather high positive correlation between the variables `n_lines` and `n_words`. The variable  `season` seems to have a moderately strong negative correlation with `total_votes`. This indicates that there could be multicollinearity relationship between our variables.  

Next, we perform a series of boxplot to visualise the relationships between `imdb_rating` and other categorical variables such as `air_month` and `char_most_lines`.




```python
sns.catplot(
    x = "air_month",
    y = "imdb_rating",
    kind = "box",
    data = d,
    aspect = 2
)
```




    <seaborn.axisgrid.FacetGrid at 0x7ffae02e30d0>




    
![png](project1_files/project1_25_1.png)
    



```python
sns.catplot(
    x = "char_most_lines",
    y = "imdb_rating",
    kind = "box",
    data = d,
    aspect = 2
)
```




    <seaborn.axisgrid.FacetGrid at 0x7ffad8142fd0>




    
![png](project1_files/project1_26_1.png)
    


The plots for other categorical features like `director`, `writer` and `main_chars` are extremely disordered and thus excluded. This is mainy due to the fact that these categorical features consist of a large number of levels, and also some errors like misspellings and duplicates of names. In the following section, we will be correcting the errors and dummy coding these categorical features.


```python
#Fixing typos in the naming of some directors
d['director']=d['director'].replace('Ken Wittingham', 'Ken Whittingham')
d['director']=d['director'].replace('Greg Daneils', 'Greg Daniels')
d['director']=d['director'].replace('Charles McDougal', 'Charles McDougall')
d['director']=d['director'].replace('Paul Lieerstein', 'Paul Lieberstein')
d['director']=d['director'].replace('Claire Scanlong', 'Claire Scanlon')

```


```python
def group_to_bool (data, column_name, role, make_list = False):
    d = data
    # make a list of main characters.
    person_list = d[column_name].str.split(';', expand=True)
    # data frame to list
    person_list = person_list.values.tolist()
    # convert to one-dimention
    person_list = sum(person_list,[])
    #  convert the given set into list
    person_list = list(set(person_list))
    # delete none value
    person_list = list(filter(None, person_list))
    
    # loop to add column with bool for checking person's exsistence
    for i in range(len(person_list)):
        # extract name for making bool
        name = person_list[i]
        # make bool by referring given column
        dd = d[column_name].str.contains(name)
        # add column to original data
        d[role+name] = dd
        # convert True & False to 1 & 0 
        d[role+name] = d[role+name].astype(int)
        # person list rewrite with role
        person_list[i] = role+name
    
    if make_list == False:
        return d
    else:
        return d, person_list
```

### Dummy coding the categorical features: `director`, `writer`, `main_chars`, `char_most_lines`, `air_month`

Now, we will perform feature engineering on the categorical variables by using dummy coding.


```python
d, writercol = group_to_bool(d, 'writer', 'writer ', make_list = True)
d, directorcols = group_to_bool(d, 'director', 'director ' ,make_list = True)
d = group_to_bool(d, 'main_chars', 'main_chars ')
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
      <th>season</th>
      <th>episode_name</th>
      <th>director</th>
      <th>writer</th>
      <th>imdb_rating</th>
      <th>total_votes</th>
      <th>air_month</th>
      <th>n_lines</th>
      <th>n_words</th>
      <th>main_chars</th>
      <th>...</th>
      <th>main_chars Angela</th>
      <th>main_chars Ryan</th>
      <th>main_chars Michael</th>
      <th>main_chars Kevin</th>
      <th>main_chars Dwight</th>
      <th>main_chars Stanley</th>
      <th>main_chars Meredith</th>
      <th>main_chars Erin</th>
      <th>main_chars Kelly</th>
      <th>main_chars Oscar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Pilot</td>
      <td>Ken Kwapis</td>
      <td>Ricky Gervais;Stephen Merchant;Greg Daniels</td>
      <td>7.6</td>
      <td>8.217708</td>
      <td>Mar</td>
      <td>5.433722</td>
      <td>7.921898</td>
      <td>Angela;Dwight;Jim;Kevin;Michael;Oscar;Pam;Phyl...</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Diversity Day</td>
      <td>Ken Kwapis</td>
      <td>B.J. Novak</td>
      <td>8.3</td>
      <td>8.179200</td>
      <td>Mar</td>
      <td>5.313206</td>
      <td>7.940228</td>
      <td>Angela;Dwight;Jim;Kelly;Kevin;Michael;Oscar;Pa...</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Health Care</td>
      <td>Ken Whittingham</td>
      <td>Paul Lieberstein</td>
      <td>7.9</td>
      <td>8.000685</td>
      <td>Apr</td>
      <td>5.497168</td>
      <td>7.926242</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>The Alliance</td>
      <td>Bryan Gordon</td>
      <td>Michael Schur</td>
      <td>8.1</td>
      <td>7.967627</td>
      <td>Apr</td>
      <td>5.493061</td>
      <td>7.985825</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Basketball</td>
      <td>Greg Daniels</td>
      <td>Greg Daniels</td>
      <td>8.4</td>
      <td>8.064322</td>
      <td>Apr</td>
      <td>5.438079</td>
      <td>7.798523</td>
      <td>Angela;Darryl;Dwight;Jim;Kevin;Michael;Oscar;P...</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>



The resulting dataframe has expanded to 123 columns mainly due to a large number of writers and directors. This is not ideal as this number is almost as big as the number of observations we have (186), which may lead to the Curse of Dimensionality. Therefore, instead of dummy coding every writer, we group the writers according to the frequency they appear in the data set:

`writer_high`: writers that have written 15 or more episodes; `writer_mid`: writers that have written 10 or more but less than 15 episodes'; `writer_midlow`: writers that have written 5 or more but less than 10 episodes; `writer_low`: writers that have written less than 5 episodes

As for directors, we keep the top 10 directors based on the number of episodes they have directed, and group the remaining directors in the `other_directors` category. 



```python
# Create empty categories in order to be filled in later by the number of writers who satisfy a certain conditions. 
writer_high=[]
writer_mid=[]
writer_midlow=[]
writer_low=[]

for i in writercols:
    if (d[i].sum() >= 15):
        writer_high.append(i)
    elif (d[i].sum()>= 10 and d[i].sum() < 15):
        writer_mid.append(i)
    elif (d[i].sum()>= 5 and d[i].sum()< 10):
        writer_midlow.append(i)
    else:
        writer_low.append(i)

d['writer_high']=d[writer_high].sum(axis=1)
d['writer_mid']=d[writer_mid].sum(axis=1)
d['writer_midlow']=d[writer_midlow].sum(axis=1)
d['writer_low']=d[writer_midlow].sum(axis=1)

writerlevel=['writer_high', 'writer_mid', 'writer_midlow', 'writer_low']
for k in writerlevel:
    dd=d[k]>0
    d[k]=dd
    d[k]=d[k].astype(int)
d=d.drop(writercols, axis=1)
```


```python
other_directors=list((d[directorcols].sum()).sort_values(ascending=False).index[11:])
d['other_directors']= d[other_directors].sum(axis=1)
dd=d['other_directors']>0
d['other_directors']=dd
d['other_directors']=d['other_directors'].astype(int)

d=d.drop(other_directors, axis=1)
```

We now dummy code the rest of the categorical variables (`air_month`, `season` and `char_most_lines`). We will also be excluding the `episode_name` variable from our model since we do not find it that useful to predict IMDb ratings from the names of episodes.


```python
d=pd.get_dummies(d, columns=['air_month','char_most_lines','season'])
d=d.drop(['director','writer','episode_name','main_chars'], axis=1)
#d.head(5)
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
      <th>imdb_rating</th>
      <th>total_votes</th>
      <th>n_lines</th>
      <th>n_words</th>
      <th>director Randall Einhorn</th>
      <th>director Paul Feig</th>
      <th>director Matt Sohn</th>
      <th>director Greg Daniels</th>
      <th>director David Rogers</th>
      <th>director Paul Lieberstein</th>
      <th>...</th>
      <th>char_most_lines_Pam</th>
      <th>season_1</th>
      <th>season_2</th>
      <th>season_3</th>
      <th>season_4</th>
      <th>season_5</th>
      <th>season_6</th>
      <th>season_7</th>
      <th>season_8</th>
      <th>season_9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.6</td>
      <td>8.217708</td>
      <td>5.433722</td>
      <td>7.921898</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3</td>
      <td>8.179200</td>
      <td>5.313206</td>
      <td>7.940228</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.9</td>
      <td>8.000685</td>
      <td>5.497168</td>
      <td>7.926242</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.1</td>
      <td>7.967627</td>
      <td>5.493061</td>
      <td>7.985825</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.4</td>
      <td>8.064322</td>
      <td>5.438079</td>
      <td>7.798523</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 64 columns</p>
</div>



## 3. Model Fitting and Tuning

We wil be using the helper functions provided in Workshop 5, namely `get_coefs()` and `model_fit()`:


```python
#Helper function obtained from Workshop 5 
from sklearn.metrics import mean_squared_error
def get_coefs(m):
    # If pipeline, use the last step as the model
    if (isinstance(m, sklearn.pipeline.Pipeline)):
        m = m.steps[-1][1]
    
    
    if m.intercept_ is None:
        return m.coef_
    return np.concatenate([[m.intercept_], m.coef_])

def model_fit(m, X, y, plot = False):
    y_hat = m.predict(X)
    rmse = mean_squared_error(y, y_hat, squared=False)
    
    res = pd.DataFrame(
        data = {'y': y, 'y_hat': y_hat, 'resid': y - y_hat}
    )
    
    if plot:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.lineplot(x='y', y='y_hat', color="grey", data =  pd.DataFrame(data={'y': [min(y),max(y)], 'y_hat': [min(y),max(y)]}))
        sns.scatterplot(x='y', y='y_hat', data=res).set_title("Fit plot")
        
        plt.subplot(122)
        sns.residplot(x='y', y='resid', data=res, lowess=True, color="g", label= "Residual plot")
        #sns.scatterplot(x='y', y='resid', data=res).set_title("Residual plot")
        
        
        plt.subplots_adjust(left=0.0)
        
        plt.suptitle("Model rmse = " + str(round(rmse, 4)), fontsize=16)
        plt.show()
    
    return rmse
```

Before we begin fitting the model, we split the data into training (X_train, y_train) and test sets (X_test, y_test) using a 70/30 split. This results in the training set having 130 observations, leaving 56 observations for the test set.


```python
#Split data into training and test sets
y=d.imdb_rating
X=d.drop(columns=['imdb_rating'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)
```


```python
# A. Using Linear Regression Model as Baseline Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=False).fit(X_train, y_train)
model_fit(lm, X_test, y_test)

#linear model score on traiing data and test data.
print("Regression score on test data is", lm.score(X_test, y_test))
print("Regression score on train is", lm.score(X_train, y_train))
```


    
![png](project1_files/project1_43_0.png)
    


    Regression score on test data is 0.4076002703688347
    Regression score on train is 0.8863340652138891


Based one the regression score, the current model overfit the training data and underfit the testing data. As a result, it will not perform well when given a new data set. Thus, it is advisable that we do not use the model above for predicting `imdb_ratings`. 

After trying out a few models including linear regression, polynomial regression (looping over different degrees of the numerical variables without any interactions), Ridge regression, Kernel Ridge regression and Lasso regression, we decided that our model of choice would be the Lasso regression. As mentioned previously in the section for exploratory data analysis, there may be issues relating to multicollinearity. Therefore, using regularization (either Ridge or Lasso) could help alleviate this issue. We were also concerned with the problems of overfitting or underfitting. If our model is too simple and has very few parameters then it may have high bias and low variance (underfitting). On the other hand, if our model has a large number of parameters then it is going to have high variance and low bias (overfitting). It is not possible to achieve a model with low variance and low bias. In fact, we need to find a good balance between bias and variance (bias-variance tradeoff). By choosing a regularization method, we are able to decrease their variance at the cost of increasing bias.

Moreover, despite our earlier efforts in reducing the number of features in the previous section (63 in total), the total number of features is still quite large, especially when comparing it against the number of observations for the training set (130). We wish to further narrow down the number of features to enhance interpretability by using a regression analysis method that not only performs regularization, but also variable selection. In this case, the Lasso meets these requirements. 


We first try to fit a Lasso regression with the tuning parameter `alpha` = 0.20 (chosen arbitrarily) which determines the weight of the  $\ell_2$ penalty. After comparing the results between Lasso regression models with and  without standardizing the numerical variables, we obtain lower root mean squared error (rmse) for the model with standardization. Also, it is recommended that all features are standardised before fitting using Lasso regression so that the solution does not depend on the measurement scale. The standardization step is thus included in the pipeline (`StandardScaler()`).


```python
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler

l = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.20)
).fit(X_train, y_train)

```

We obtain the following results:


```python
#print rmse and display fit plot and residual plot
print("lasso rmse:", model_fit(l, X_test, y_test, plot=True))
```


    
![png](project1_files/project1_49_0.png)
    


    lasso rmse: 0.3772125190659439





    {'total_votes': (0.1813738719317472,),
     'n_lines': (0.0,),
     'n_words': (0.0,),
     'director Randall Einhorn': (-0.0,),
     'director Paul Feig': (0.0,),
     'director Matt Sohn': (-0.0,),
     'director Greg Daniels': (0.0,),
     'director David Rogers': (-0.0,),
     'director Paul Lieberstein': (-0.0,),
     'director Ken Kwapis': (0.0,),
     'director B.J. Novak': (-0.0,),
     'director Jeffrey Blitz': (-0.0,),
     'director Ken Whittingham': (0.0,),
     'director Charles McDougall': (0.0,),
     'main_chars Andy': (0.0,),
     'main_chars Phyllis': (0.0,),
     'main_chars Toby': (0.0,),
     'main_chars Creed': (0.0,),
     'main_chars Darryl': (-0.0,),
     'main_chars Jim': (0.0,),
     'main_chars Pam': (0.0,),
     'main_chars Angela': (-0.0,),
     'main_chars Ryan': (-0.0,),
     'main_chars Michael': (0.0,),
     'main_chars Kevin': (-0.0,),
     'main_chars Dwight': (0.0,),
     'main_chars Stanley': (0.0,),
     'main_chars Meredith': (0.0,),
     'main_chars Erin': (-0.0,),
     'main_chars Kelly': (0.0,),
     'main_chars Oscar': (-0.0,),
     'writer_high': (0.0,),
     'writer_mid': (0.0,),
     'writer_midlow': (-0.0,),
     'writer_low': (-0.0,),
     'other_directors': (-0.0,),
     'air_month_Apr': (-0.0,),
     'air_month_Dec': (0.0,),
     'air_month_Feb': (0.0,),
     'air_month_Jan': (-0.0,),
     'air_month_Mar': (-0.0,),
     'air_month_May': (0.0,),
     'air_month_Nov': (-0.0,),
     'air_month_Oct': (-0.0,),
     'air_month_Sep': (-0.0,),
     'char_most_lines_Andy': (-0.0,),
     'char_most_lines_Deangelo': (-0.0,),
     'char_most_lines_Dwight': (-0.0,),
     'char_most_lines_Erin': (-0.0,),
     'char_most_lines_Jim': (0.0,),
     'char_most_lines_Jo': (-0.0,),
     'char_most_lines_Michael': (0.0,),
     'char_most_lines_Nellie': (-0.0,),
     'char_most_lines_Pam': (-0.0,),
     'season_1': (-0.0,),
     'season_2': (-0.0,),
     'season_3': (0.0,),
     'season_4': (0.0,),
     'season_5': (0.0,),
     'season_6': (-0.0,),
     'season_7': (0.0,),
     'season_8': (-0.0,),
     'season_9': (-0.0,)}



It is obvious from the fit plot that the model does not perform well at all- the points do not align with the straigh line at all. From the residual plot, we see that the model tends to overestimate for lower values of IMDb ratings and underestimate for higher values of IMDb ratings. As for the estimated coefficients, all but the coefficient for `total_votes` are shrunk to 0, this model fails to be informative. Perhaps we can tune the `alpha` parameter to find the most optimal Lasso regression model. We do this using GridSearchCv.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


alphas = np.linspace(0.01, 1, num=100)

l_gs = GridSearchCV(
    make_pipeline(
        StandardScaler(),
        Lasso()
    ),
    param_grid={'lasso__alpha': alphas},
    cv=KFold(5, shuffle=True, random_state=1234),
    scoring="neg_root_mean_squared_error"
).fit(X_train, y_train)
```

We obtain the following results using GridSearchCv:


```python
print( "best tuning alpha:", l_gs.best_params_['lasso__alpha'])
print( "best rmse :", l_gs.best_score_ * -1)
```

    best tuning alpha: 0.03
    best rmse : 0.3189933226278793


Our final model is therefore the Lasso regression model with tuning parameter `alpha` = 0.01. The coefficients for this model are given below:


```python
l_gs_coefs = get_coefs(l_gs.best_estimator_)[1:]
dict(zip(X.columns, zip(l_gs_coefs)))
```




    {'total_votes': (0.3504787363836297,),
     'n_lines': (0.07692901536906813,),
     'n_words': (0.0,),
     'director Randall Einhorn': (-0.0,),
     'director Paul Feig': (0.0,),
     'director Matt Sohn': (-0.0,),
     'director Greg Daniels': (0.0,),
     'director David Rogers': (-0.0,),
     'director Paul Lieberstein': (-0.0,),
     'director Ken Kwapis': (-0.0,),
     'director B.J. Novak': (0.0,),
     'director Jeffrey Blitz': (-0.0,),
     'director Ken Whittingham': (-0.0,),
     'director Charles McDougall': (0.016724648450323377,),
     'main_chars Andy': (0.00920145891696371,),
     'main_chars Phyllis': (-0.0,),
     'main_chars Toby': (0.0,),
     'main_chars Creed': (0.0,),
     'main_chars Darryl': (-0.0,),
     'main_chars Jim': (0.08844698339293966,),
     'main_chars Pam': (-0.0,),
     'main_chars Angela': (-0.0,),
     'main_chars Ryan': (-0.0,),
     'main_chars Michael': (0.0,),
     'main_chars Kevin': (-0.0,),
     'main_chars Dwight': (0.0,),
     'main_chars Stanley': (0.0,),
     'main_chars Meredith': (-0.0,),
     'main_chars Erin': (0.0,),
     'main_chars Kelly': (0.02775308927459676,),
     'main_chars Oscar': (0.0,),
     'writer_high': (0.0,),
     'writer_mid': (0.0,),
     'writer_midlow': (0.0,),
     'writer_low': (0.0,),
     'other_directors': (0.0,),
     'air_month_Apr': (-0.0,),
     'air_month_Dec': (0.011686697896927223,),
     'air_month_Feb': (0.0,),
     'air_month_Jan': (0.0,),
     'air_month_Mar': (-0.0,),
     'air_month_May': (0.047144941405997745,),
     'air_month_Nov': (-0.0,),
     'air_month_Oct': (-0.012162698507229692,),
     'air_month_Sep': (-0.0,),
     'char_most_lines_Andy': (-0.006450275104394255,),
     'char_most_lines_Deangelo': (-0.0,),
     'char_most_lines_Dwight': (0.0,),
     'char_most_lines_Erin': (-0.0,),
     'char_most_lines_Jim': (-0.0,),
     'char_most_lines_Jo': (-0.0,),
     'char_most_lines_Michael': (0.0,),
     'char_most_lines_Nellie': (-0.0645935435995787,),
     'char_most_lines_Pam': (0.0,),
     'season_1': (-0.10996877153237057,),
     'season_2': (-0.03365638094324081,),
     'season_3': (0.0,),
     'season_4': (0.0,),
     'season_5': (0.02591353253872528,),
     'season_6': (-0.0,),
     'season_7': (0.012359377963924447,),
     'season_8': (-0.07443874087320965,),
     'season_9': (-0.0,)}



The rmse for the model is given as:


```python
model_fit(l_gs.best_estimator_, X_test, y_test, plot=True)

# Model score after regularization and GridSearch
print("Lasso score on test data is", round(l_gs.best_estimator_.score(X_test, y_test),3))
print("Lasso score on train data is", round(l_gs.best_estimator_.score(X_train, y_train),3))
```


    
![png](project1_files/project1_57_0.png)
    


    Lasso score on test data is 0.675
    Lasso score on train data is 0.794


Based on the model scores, the Lasso model with the optimized alpha (through GridSearch) performs better than the regular linear model above.  
We could also assess how far the true generalization error might deviate from the average test error, which is indicated by Average test error±2(standard error of average test error)Average \ test \ error \pm 2(standard \ error \ of\ average\ test\ error)Average test error±2(standard error of average test error). We use the squared error loss for our model, in this case, the average test error is just the mean squared error.


```python
#Compute the average test error
y_hat = l_gs.best_estimator_.predict(X_test)
avg_test_error = mean_squared_error(y_test, y_hat)

#Compute the standard error of the average test error
squared_error = (y_hat-np.array(y_test))**2
se_average_test_error = np.sqrt(np.var(squared_error,ddof=1)/len(squared_error)) 
#ddof=1,the sample variance is used to estimate the variance of the test errors

print("The average test error is ", round(avg_test_error,4))
print("The standard error of the average test error is ", round(se_average_test_error,4))
print("The true generalization error might deviate from the average test error by a magnitude of approximately ", round(2*se_average_test_error,4))
```

    The average test error is  0.0816
    The standard error of the average test error is  0.0152
    The true generalization error might deviate from the average test error by a magnitude of approximately  0.0304


## 4. Discussion & Conclusions


## Overview of the final model



For our final model, we predict IMDb rating of a given episode using the following features:
- `total_votes`: Number of ratings for episode on IMDb      
- `n_lines`: Number of spoken lines in episode
- `n_words`: Number of dialog words in episide 
- `directors`: Episode directors (we pick the top 10 directors based on their experience directing the most number of episodes)
- `writers`: Episode writers (we split the writers into 4 groups based on the number of episodes they wrote)
- `main_chars`: Main characters appearing in episode (main characters were determined to be characters appearing in more than 1/2 of the episodes)
- `season`: Season number of the episode
- `air_month` : The month the episode was aired
- `char_most_lines`: Characters with most lines in the episode

For each feature, we obtain a 'weight' value, which we can interpret as the relative importance of how each feature affects the IMDb rating. We will be giving recommendations based on these weights. 

As high dimensionality may cause an increase in test error and in the raw data we have noisy features which are not truly associated with the popularity or the IMDb ratings, we have done feature engineering in order to analyse the data which we think are truly associated with the ratings. Also, Lasso helps in feature selection which is very useful for analysing this dataset while Rdige regression cannot perform this kind of selection. Therefore, we could ensure that the reliability of the model would not be affected by the noisy features. 

As we observe from the weights of the inputs, most of the weights are close to 0 while just very few of them have more substantial weights. This is one of the reasons why we choose Lasso. In order to avoid overfitting, Lasso helps us eliminating irrelevant variables which are irrelevant to the IMDb ratings by the tuning parameters $\alpha$. This also improves the interpretability and reduces the complexity of the model.

Finally, we have assessed how accurate our model is able to predict outcome values for previously unseen data, which is indicated by the average test error. We obtain an average test error of approximately 0.0816, which is fairly acceptable. We also assessed how the true accuracy would deviate from our current accuracy, and we obtained a low deviation of approximately 0.0152. This means that for our model the highest average test error we may get is $0.0816+0.0304=0.112$ 


## Recommendations

From the model, we observe that the number of spoken lines in an episode has a small positive effect on the IMDb rating, therefore a script with more lines may contribute to the IMDb ratings slightly. Among the top 10 directors, only Charles McDougall is associated with positive weight, that means choosing him to direct the reunion episode may have a slight positive effect on the IMDb rating. As for the writers, we observe that all 4 categories (i.e. `writer_high` (wrote 15 or more episodes), `writer_mid` (wrote 10 or more but less than 15 episodes), `writer_midlow` (wrote 5 or more but less than 10 episodes) and `writer_low` (wrote less than 5 episodes)) have weights 0, meaning that choosing writers from either category will not affect the IMDb rating significantly. However, it may be sensible to consider writers who are more experienced. 

Since it is a reunion episode, it is preferable to include all main characters. However to improve the ratings, our model have suggested that we should only give a few characters with the most screen time. The main characters that are associated with positive weights are Jim, Kelly and Andy, with Jim having the highest weight followed by Kelly then Andy; this means increasing the screen time for these characters may increase IMDb ratings. Furthermore, by looking at the characters with most lines per episode, we can deduce that Andy and Nellie are associated with negative weights, with Nellie's weight being much more negative than Andy's. Hence, if Nellie appears as the character who spoken the most lines, this could potentially lead to a decrease in the episode ratings. Eventhough, Andy's weight is negative, its magnitude is relatively small; so giving him more lines won't affect the rating significantly. Additionally, since the associated weights are 0 for other characters such as (Deangelo, Dwight, Erin, Jim, Jo, Michael and Pam); it is also safe to give more lines to those other characters. 

By looking at how each season effects the IMDb rating using our model, we observe that season 1, season 2 and season 8 are associated with negative weights, with season 1 having the most negative impact on IMDb rating followed by season 8 then 2. In contrast to this, seasons 5 and 7 are associated with positive weights, with season 5 being more positive than season 7. With this information, one can examine how the negatively weighted seasons (1,2 and 8) differ from the positively weighted seasons (5 and 7). Perhaps season 5 and 7 are characterised by elements like characters, writing, plot, etc. that viewers may prefer over some elements that are found in season 1,2 and 8. 

Another feature that we look at was the month that the episodes were aired. Our results indicate that months like May and December have positive weights, while October has negative weight. Similar to season, one can compare how certain elements differ between episodes aired in May/December and episodes aired in October. From this, we can infer that there could be a possibility that viewers enjoy festive/Christmas themed that are aired in December. 

Finally, we observe that the total votes have the biggest positive effect on IMDb rating relative to all other features. Its effect may be 10 times as big as some of the features. At the end of the day, the number of ratings contributes the most to the IMDb rating therefore it is important to ensure that there is a sufficient number of voters. This may be an issue since the show has seen a drop in number of ratings throughout each season, which may indicate that people are losing interest. Therefore, it is essential that the reunion episode gains a large audience and this may be achieved by employing good promotion or marketing strategy prior to the episode's release (releasing teaser, using social media platforms, advertisements etc.) to increase anticipation. In addition, increaing the number of platforms (streaming services) whrereby the reunion episode (or even the whole show) is aired may also expand the fan base.

## Limitations
It is difficult to determine the effects from the individual writers and directors as there are several writers and directors for each episode and they contribute to the same episode. A similar limitation occurs at 'season' and 'episode' since not all the TV show has the same number of seasons and episodes. This may affect the accuracy of prediction. Moreover, the sample size is limited. If we have inputs variables such as names of writers or directors which are not included in this limited dataset, we could not predict the ratings based on the inputs. However, there are many writers or directors from different TV show. The model restricts the data that we could use for prediction. Additionally, the setting of a threshold such as 15 or 10 for grouping writer and director is arbitrary. We made sure that our setting is surely one of the reasonable ways for grouping in terms of the number of writer and director in each group but is still debatable. 

## 5. Convert Document


```python
# Run the following to render to PDF
!jupyter nbconvert --to markdown project1.ipynb
```

    [NbConvertApp] Converting notebook project1.ipynb to markdown
    [NbConvertApp] Support files will be in project1_files/
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Making directory project1_files
    [NbConvertApp] Writing 47548 bytes to project1.md


<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=855a35f7-30d3-46a8-952e-1f6c0ecba2f3' target="_blank">
<img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
