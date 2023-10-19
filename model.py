# %% [markdown]
# ### This notebook contains our Diabetes dataset quick exploration and our model diagnosis

# %% [markdown]
# The goal is to deploy a Logistic Regression model that can predict whether or not a person would have Diabetes given certain medical parameters, such as amount of pregnancies, level of glucose, blood pressure, skin thickness, insulin levels, BMI and age of the person.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('Healthcare-Diabetes.csv', index_col = 'Id')
df.info()

# %%
df.head()

# %%
df.describe()

# %% [markdown]
# We are eliminating the Diabetes Pedigree Function column, as it is another predictor model.

# %%
df = df.drop('DiabetesPedigreeFunction', axis = 1)

# %% [markdown]
# Let's check how balanced our dependent variabe is:

# %%
sns.countplot(x='Outcome', data=df, palette='RdBu_r')

# %% [markdown]
# The data seems to be clean. We have no null values and a suffcient sampe size.

# %% [markdown]
# ### Training and Predicting

# %%
from sklearn.model_selection import train_test_split

# %%
x_train, x_test, y_train, y_test = train_test_split(df.drop('Outcome',axis=1), 
                                                    df['Outcome'], test_size=0.30, 
                                                    random_state=101)

# %%
from sklearn.linear_model import LogisticRegression

# %%
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

# %%
predictions = logmodel.predict(x_test)

# %% [markdown]
# ### Diagnosis

# %%
from sklearn.metrics import classification_report

# %%
print(classification_report(y_test,predictions))

# %% [markdown]
# Finally, we are going to save our model

# %%
import pickle

# %%
pickle.dump(logmodel, open('model.pkl','wb'))

# %% [markdown]
# **Note: No in-depth EDA was performed as that was not the purpose of this project. Instead, a model was built and deployed using Flask.*


