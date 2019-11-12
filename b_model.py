import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle



# Import cleaned data 
updated_stats = pd.read_csv("nba_stats3.csv")

# assigne X, y
target = updated_stats['All_NBA'].values.reshape(-1, 1)
data = updated_stats.drop("All_NBA", axis=1)

# Split X, y 
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)

# Scale X, y
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
y_train_scaled = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)



#LASSO model
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=.01).fit(X_train_scaled, y_train_scaled)

training_score1 = lasso.score(X_train_scaled, y_train_scaled)
testing_score1 = lasso.score(X_test_scaled, y_test_scaled)

predictions1 = lasso.predict(X_test_scaled)
MSE = mean_squared_error(y_test_scaled, predictions1)
r2 = lasso.score(X_test, y_test_scaled)

print(f"MSE: {MSE}, R2: {r2}")
print(f"Training Score: {training_score1}")
print(f"Testing Score: {testing_score1}")


#saving model
with open(f'best_model.pickle', 'wb') as f:
    pickle.dump(lasso, f)


# model = LinearRegression()

# model.fit(X_train_scaled, y_train_scaled)
# training_score = model.score(X_train_scaled, y_train_scaled)
# testing_score = model.score(X_test_scaled, y_test_scaled)

# predictions = model.predict(X_test_scaled)
# MSE = mean_squared_error(y_test, predictions)
# r2 = model.score(X_test_scaled, y_test_scaled)

# print(f"MSE: {MSE}, R2: {r2}")
# print(f"Training Score: {training_score}")
# print(f"Testing Score: {testing_score}")






# # Ridge model
# # Note: Use an alpha of .01 when creating the model for this activity
# from sklearn.linear_model import Ridge

# ridge = Ridge(alpha=.01).fit(X_train_scaled, y_train_scaled)

# training_score2 = ridge.score(X_train_scaled, y_train_scaled)
# testing_score2 = ridge.score(X_test_scaled, y_test_scaled)

# predictions2 = ridge.predict(X_test_scaled)
# MSE = mean_squared_error(y_test_scaled, predictions2)
# r2 = ridge.score(X_test_scaled, y_test_scaled)

# print(f"MSE: {MSE}, R2: {r2}")
# print(f"Training Score: {training_score2}")
# print(f"Testing Score: {testing_score2}")