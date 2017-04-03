import tensorflow.contrib.learn as learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()

x = preprocessing.StandardScaler().fit_transform(boston.data)
feature_columns = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, boston.target, steps=200, batch_size=32)
boston_predictions = list(regressor.predict(x, as_iterable=True))

print ' \n--- EXAMPLE DATA ---\n'
print boston.data
print ' \n--- EXPECTED DATA RESULTS ---\n'
print boston.target
print ' \n--- PREDICTED DATA RESULTS ---\n'
print boston_predictions

score = metrics.mean_squared_error(boston_predictions, boston.target)

print ' \n--- MSE: ---\n'
print score
