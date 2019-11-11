#Import project libraries
import numpy as np
import pandas as pd
import pickle

#Reading data file
dataset = pd.read_csv('loyalty.csv')
df = dataset

#Cleaning data set
#Removing all rows with ERROR in Year_2_Revenue
df = df[df.Year_2_Revenue != 'ERROR']

#Removing $' and ',' on the Revenue columns with acustom function
def replace_currency(x):
    try: 
        return float(x.replace('$', '').replace(',',''))
    except AttributeError:
        return np.NaN
#Applying the replace_currency()
df['Year_0_Revenue'] = df['Year_0_Revenue'].apply(replace_currency)
df['Year_1_Revenue'] = df['Year_1_Revenue'].apply(replace_currency)
df['Year_2_Revenue'] = df['Year_2_Revenue'].apply(replace_currency)
df['Year_3_Revenue'] = df['Year_3_Revenue'].apply(replace_currency)
df['Year_4_Revenue'] = df['Year_4_Revenue'].apply(replace_currency)

#Replace all blanks with zeros
df = df.fillna(0)

#Dropping contract_id and Join_Date
df = df.drop(['Contractor_ID','Join_Date'], axis=1)

#Creating dummy variables for all our categorical features and saving results to a new dataframe 'abt'-Analytical Base Table
abt = pd.get_dummies(df, columns=['Contractor_Loyalty_Status', 'Region'])

#Write abt to csv
abt.to_csv('loyalty_clean.csv', index = None)

# Loading ABT
df = pd.read_csv('loyalty_clean.csv')

#Modeling
# 1. importing Tree Ensamble algo
from sklearn.ensemble import GradientBoostingRegressor

# Importing function for splitting training and test set
from sklearn.model_selection import train_test_split

# Create separate object for target variable
y = df.Year_4_Revenue

# Create separate object for input features
X = df.drop('Year_4_Revenue', axis=1)

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=1234)

# 2. Preprocessing & Pipelines
# Standardize X_train
X_train_new = (X_train - X_train.mean()) / X_train.std()

#preprocess new, unseen data (such as our test set) in the same way
X_test_new = (X_test - X_train.mean()) / X_train.std()

#Creating Pipeline
# Importing lib for creating model pipelines
from sklearn.pipeline import make_pipeline

# Importing lib For standardization
from sklearn.preprocessing import StandardScaler

# Create pipelines dictionary
pipelines = {
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}

# Hyperparameter tuning
#Setting (Declaring)a hyperparameter grid for GradientBoostingRegressor
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators': [100,200],
    'gradientboostingregressor__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}

# Create hyperparameters dictionary
hyperparameters = {
    'gb' : gb_hyperparameters
}


#3. 10-Fold Cross-Validation (CV)
# Import a helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Create cross-validation object from gb pipeline and gb hyperparameters
model = GridSearchCV(pipelines['gb'], hyperparameters['gb'], cv=10, n_jobs=-1)

# To ignore ConvergenceWarning messages
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

#4. Model fitting and Tuning
# Fit and tune model
model.fit(X_train, y_train)

# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    #Print '{name} has been fitted'
    print(name, 'has been fitted.')

# 5. Model performance
#Importing mean_absolute_error
from sklearn.metrics import mean_absolute_error

# Predict test set using fitted random forest
pred = fitted_models['gb'].predict(X_test)


from sklearn.metrics import r2_score

# Calculate and print R^2 and MAE
print( 'R^2:', r2_score(y_test, pred ))
print( 'MAE:', mean_absolute_error(y_test, pred))


#Saving model to disk as a pkl file
pickle.dump(model , open('model.pkl','wb'))

#Loading model to compare results
model = pickle.load(open('model.pkl','rb'))
