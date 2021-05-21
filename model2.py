import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train_data = pd.read_csv("train.csv")
train_labels = train_data['Survived']

columns_to_extract = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

train_features = train_data[columns_to_extract]

train_features = pd.concat([train_features, pd.get_dummies(train_features['Pclass'],prefix='Pclass')], axis=1)
train_features.drop(['Pclass'],axis=1,inplace=True)

train_features = pd.concat([train_features, pd.get_dummies(train_features['Embarked'],prefix='Embarked')], axis=1)
train_features.drop(['Embarked'],axis=1,inplace=True)

train_features = pd.concat([train_features, pd.get_dummies(train_features['Sex'],prefix='Sex')], axis=1)
train_features.drop(['Sex'],axis=1,inplace=True)

train_features['Cabin'] = train_features['Cabin'].fillna('C')
train_features['Cabin1'] = train_features['Cabin'].astype(str).str[0]

train_features = pd.concat([train_features, pd.get_dummies(train_features['Cabin1'],prefix='Cabin1')], axis=1)
train_features.drop(['Cabin'],axis=1,inplace=True)
train_features.drop(['Cabin1'],axis=1,inplace=True)
train_features.drop(['Cabin1_T'],axis=1,inplace=True)

print(train_features.info())

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer((None,17,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(train_features, train_labels, validation_split=0.2, epochs=100)

test_loss, test_accuracy = model.evaluate(train_features, train_labels, verbose=2)
print (test_loss)
print (test_accuracy)

print(train_features.head())

test_data = pd.read_csv("test.csv")

test_features= test_data[columns_to_extract]

test_features = pd.concat([test_features, pd.get_dummies(test_features['Pclass'],prefix='Pclass')], axis=1)
test_features.drop(['Pclass'],axis=1,inplace=True)

test_features = pd.concat([test_features, pd.get_dummies(test_features['Embarked'],prefix='Embarked')], axis=1)
test_features.drop(['Embarked'],axis=1,inplace=True)

test_features = pd.concat([test_features, pd.get_dummies(test_features['Sex'],prefix='Sex')], axis=1)
test_features.drop(['Sex'],axis=1,inplace=True)
test_features['Cabin'] = test_features['Cabin'].fillna('C')
test_features['Cabin1'] = test_features['Cabin'].astype(str).str[0]

test_features = pd.concat([test_features, pd.get_dummies(test_features['Cabin1'],prefix='Cabin1')], axis=1)
test_features.drop(['Cabin'],axis=1,inplace=True)
test_features.drop(['Cabin1'],axis=1,inplace=True)

print(test_features.head())

predicciones = model.predict(test_features)
out = predicciones.round().astype(int)

dataset_envio = test_data['PassengerId']
dataset_envio = pd.concat([dataset_envio, pd.DataFrame(data=out, columns=['Survived'])], axis=1)

print(dataset_envio.head())

dataset_envio.to_csv('model2.csv', index=False)



