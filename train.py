import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# 1. Load Data
df = pd.read_csv('stud.csv')
X = df.drop(columns=['math_score'])
y = df['math_score']

# 2. Preprocessing Pipeline
cat_features = X.select_dtypes(include=['object', 'string']).columns
num_features = X.select_dtypes(exclude=['object', 'string']).columns

preprocessor = ColumnTransformer([
    ("OneHotEncoder", OneHotEncoder(), cat_features),
    ("StandardScaler", StandardScaler(), num_features),
])

# 3. Train Model
X_prep = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Save Artifacts
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and artifacts saved!")