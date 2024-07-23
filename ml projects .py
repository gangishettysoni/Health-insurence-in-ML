#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd



# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[97]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


dataset = pd.read_excel("Health_insurance_cost (1).xlsx")


# In[4]:


dataset


# In[5]:


pd.__version__


# In[6]:


# Display the first few rows of the dataset
#print("First few rows of the dataset:")
print(dataset.head())


# In[7]:


#loading the data set we can check the first rows and columns of dataset :
dataset.tail(5)


# In[8]:


# Summary statistics
print("\nSummary statistics:")
print(dataset.describe())


# In[82]:


# Check for missing values
print("\nMissing values:")
print(dataset.isnull().sum())


# In[10]:


get_ipython().system('pip install seaborn')


# In[11]:


pip install pandas seaborn matplotlib


# In[50]:


import pandas as pd


# In[51]:


dataset = pd.read_excel("Health_insurance_cost (1).xlsx")


# In[52]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[53]:


# Exploratory Data Analysis (EDA)
sns.set(style="ticks", color_codes=True)
pairplot = sns.pairplot(dataset)


# In[54]:


pairplot.fig.set_size_inches(12, 10)


# In[55]:


# Adjusting layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 1])  # Add rect parameter to control layout boundaries


# In[56]:


# Display the plot
plt.show()


# In[57]:


import seaborn as sns


# In[88]:


# Identify categorical columns
categorical_columns = dataset.select_dtypes(include=['object']).columns

# Perform one-hot encoding
dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns)

# Now create the correlation matrix
correlation_matrix = dataset_encoded.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)

# Add a title
plt.title('Correlation Matrix')

# Show the plot
plt.show()


# In[73]:


print(dataset.columns)


# In[78]:


# Preparing the data for modeling
X = dataset.drop(columns=['health_insurance_price'])
y = dataset['health_insurance_price']


# In[75]:


print(dataset.head())


# In[76]:


print(dataset.info())


# In[79]:


# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)


# In[99]:


# Assuming you've already loaded your dataset
# dataset = pd.read_excel('Health_insurance_cost (1).xlsx')

# Check for missing values
print(dataset.isnull().sum())

# Separate features and target
X = dataset.drop(columns=['health_insurance_price '])  # Adjust 'insurance_cost' to your actual target column name
y = dataset['health_insurance_price ']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Build the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# If you want to see the coefficients of your model
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

print(f"Intercept: {model.intercept_}")


# In[90]:


# Print column names
print(dataset.columns)

# Print first few rows
print(dataset.head())

# Identify the correct target column name (replace 'charges' with the actual name)
target_column = 'charges'

# Separate features and target
X = dataset.drop(columns=[target_column])
y = dataset[target_column]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing steps
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# In[107]:


# Print column names
print(dataset.columns)

# Print first few rows
print(dataset.head())

# Identify the correct target column name
target_column = 'health_insurance_price'

# Separate features and target
X = dataset.drop(columns=[target_column])
y = dataset[target_column]



# Impute missing values in the target variable (y)
imputer = SimpleImputer(strategy='mean')  # Adjust strategy as needed
y = imputer.fit_transform(y.values.reshape(-1, 1))  # Reshape for imputation


# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing steps
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# In[106]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
#model.fit(X_train, y_train)

# Make predictions and store them in y_pred
#y_pred = model.predict(X_test)

# Visualizing the results
plt.figure(figsize=(10, 6))
#plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)  # Diagonal line
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.show()


# In[ ]:




