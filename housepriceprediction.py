import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = {
    'area': [1000, 1500, 2000, 1300, 1800],
    'bedrooms': [2, 3, 4, 2, 4],
    'age': [10, 5, 3, 20, 15],
    'price': [300000, 400000, 500000, 320000, 480000]
}
df = pd.DataFrame(data)
X = df[['area', 'bedrooms', 'age']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions:", predictions)
print("Actual:", y_test.values)