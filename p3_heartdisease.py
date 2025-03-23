import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data=pd.read_csv("lc_march25.csv")
print(data.isnull().sum())

features= data.drop(["LUNG_CANCER"], axis=1 )
target = data["LUNG_CANCER"]

nfeatures=pd.get_dummies(features)
mms= MinMaxScaler()
sfeatures = mms.fit_transform(nfeatures.values)

x_train, x_test, y_train, y_test= train_test_split(sfeatures, target)

k=int(len(data)**0.5)
if k%2==0:
	k=k+1
print(len(data), k)

model=KNeighborsClassifier(n_neighbors=k, metric="euclidean")
model.fit(x_train, y_train)

cr=classification_report(y_test, model.predict(x_test))
print(cr)
#ndarray-->pandas DataFrame--> to_csv()
fd=pd.DataFrame(sfeatures)
fd.to_csv("f1.csv")
d= [[0.575757576,	0,	0,	0,	1,	0,	1,	0,	1,	0,	1,	1,	0,	1,	1,	0 ]]
res= model.predict(d)
print(res)
