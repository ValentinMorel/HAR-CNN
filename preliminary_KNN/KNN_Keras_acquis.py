import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import pipeline

seed = 7
numpy.random.seed(seed)

dataframe = pd.read_csv("C:/Python35/acquis_smartphone/dataset_class_3axes_118_118.txt", sep = ';', header = None)
dataset = dataframe.values
X = dataset[1:,0:11].astype(float)
Y = dataset[1:,13]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    model = Sequential()
    model.add(Dense(22, input_dim=11, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=2)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator,X, dummy_y, cv=kfold)
print("Baseline :", results.mean()*100 , results.std()*100)
