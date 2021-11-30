import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def nan_filler(data):
    '''
    also a filter
    :param data:
    :return:
    '''
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.median())
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes + 1


def missing_value_checker(data):
    '''
    validation of the data
    :param data:
    :return:
    '''
    list = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            sum = data[feature].isna().sum()

            type = data[feature].dtype

            print(f'{feature}: {sum}, type: {type}')

            list.append(feature)
    print(list)

    print(len(list))


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

missing_value_checker(test_data)

test_edited = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)  # removing the titles
train_edited = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

nan_filler(test_edited)
nan_filler(train_edited)

missing_value_checker(test_edited)

missing_value_checker(train_edited)

# train_edited.shape, test_edited.shape

test_edited.info()

train_edited.info()

X = train_edited.drop('SalePrice', axis=1)
y = train_edited['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# ====================NEURAL NETWORK====================

model = Sequential(Dense(100, input_dim=75))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer="rmsprop",  loss="MSLE", metrics=["mae"])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=100)

# pd.DataFrame(history.history).plot()
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# print(history.history)

history1 = history
pd.DataFrame(history.history["loss"]).plot()
plt.ylabel('loss')
plt.xlabel('epoch')
print(history.history)

plt.show()

pd.DataFrame(history1.history["mae"]).plot()
plt.ylabel('Mean absolute error')
plt.xlabel('epoch')
print(history.history)
plt.show()

scores = model.evaluate(X_val, y_val, verbose=1)

predictions = model.predict(test_edited)

print()
print("Predictions")
print(predictions)

output = pd.DataFrame(
{
    'Id': test_data['Id'],
    'SalePrice': np.squeeze(predictions)
})
output
# print(output)
