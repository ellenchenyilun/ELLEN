## Welcome the Elle's Sample Work

Please check out in here one Data-science related sample work of Ellen ;)


### Capital Bike Rental Analysis

The is the analysis for one bike rental company for Washington D.C.

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

![Image](src)

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ellenchenyilun/Ellen/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
