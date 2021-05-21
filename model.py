import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

url_test = "test.csv"
url_train = "train.csv"

df_test = pd.read_csv(url_test)
df_train = pd.read_csv(url_train)

print(df_test.head())
print(df_train.head())

print ("cantidad de datos prueba")
print (df_test.shape)
print ("cantidad de datos entrenamiento")
print (df_train.shape)

print ("informacion de los datos")
print (df_test.info())
print (df_train.info())

print ("datos faltantes test")
print (pd.isnull(df_test).sum())
print ("datos faltantes train")
print (pd.isnull(df_train).sum())

print ("estadisticas test")
print (df_test.describe())
print ("estadisticas train")
print (df_train.describe())


############## preprocesamiento de datos

# cambio de sexo a numerico
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)
df_test['Sex'].replace(['female','male'],[0,1],inplace=True)

# cambio datos de embarque
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

# reemplazo con los datos faltantes en edades
print (df_train["Age"].mean())
print (df_test["Age"].mean())

promedio = 30
df_train['Age'] = df_train['Age'].replace(np.nan,promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

df_train['Fare'] = df_train['Fare'].replace(np.nan,0)
df_test['Fare'] = df_test['Fare'].replace(np.nan,0)

#creamos rangos de edades y las cambiamos por numeros
# de 0 a 8, de 9 a 15, de 16 a 18... etc
bandas = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1','2','3','4','5','6','7']

df_train['Age'] = pd.cut(df_train['Age'],bandas,labels=nombres)
df_test['Age'] = pd.cut(df_test['Age'],bandas,labels=nombres)

#eliminar cabina porque tiene muchos datos faltantes
df_train.drop(['Cabin'], axis=1,inplace=True)
df_test.drop(['Cabin'], axis=1,inplace=True)

#eliminar columnas no relevantes para el analisis
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'],axis=1)

#eliminar las filas con los datos perdidos
#df_train.dropna(axis=0, how='any', inplace=True)
#df_test.dropna(axis=0, how='any', inplace=True)


print(df_test.head())
print(df_train.head())

print ("cantidad de datos prueba")
print (df_test.shape)
print ("cantidad de datos entrenamiento")
print (df_train.shape)

print ("datos faltantes test")
print (pd.isnull(df_test).sum())
print ("datos faltantes train")
print (pd.isnull(df_train).sum())

# aplicacion algoritmo ML
X = np.array(df_train.drop(['Survived'],1))
y = np.array(df_train['Survived'])

#Generar los datos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)

# regresion lineal
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)

print("precision de regresion logistica")
print(logreg.score(X_train, y_train))


#Support vector machine
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
print("precision de support vector machine")
print(svc.score(X_train, y_train))


#K neighbords
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print("precision de vecinos cercanos")
print(knn.score(X_train, y_train))

#predicciones con cada algoritmo
ids = df_test['PassengerId']
prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))
out_logreg = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_logreg})
print("Prediccion regresion logistica")
print(out_logreg.head())

prediccion_svc = svc.predict(df_test.drop('PassengerId',axis=1))
out_svc = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_svc})
print("Prediccion soporte de vectores")
print(out_svc.head())

prediccion_knn = knn.predict(df_test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({'PassengerId': ids, 'Survived': prediccion_knn})
print("Prediccion vecinos mas cercanos")
print(out_knn.head())

df_export = pd.DataFrame(out_logreg, columns=['PassengerId', 'Survived'])
df_export.to_csv(r'/home/josue/Documentos/data-science/result.csv', index=False)












