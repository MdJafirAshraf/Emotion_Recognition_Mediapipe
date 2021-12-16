import cv2
import pickle
import pandas as pd
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Face_keypoints.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train shape:')
print(X_train.shape)
print(X_test.shape)
print('\nTest Shape:')
print(y_train.shape)
print(y_test.shape)

rc = RandomForestClassifier()
model = rc.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

with open('saved_model.pkl', 'wb') as f:
    pickle.dump(model, f)

