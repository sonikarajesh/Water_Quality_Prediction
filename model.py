import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv("water_potability.csv")
df.fillna(df.mean(), inplace=True)
X = df.drop('Potability',axis=1)
Y= df['Potability']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)
dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 10, splitter= 'best')
dt.fit(X_train,Y_train)
prediction=dt.predict(X_test)
print("Accuracy Score = ",{accuracy_score(Y_test,prediction)*100})

pickle.dump(dt, open('model.pkl','wb'))# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

