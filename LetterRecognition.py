import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps
import os, ssl, time

X = np.load("dataset/image.npz")["arr_0"]
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)
# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size = 2500, train_size=7500, random_state = 9)
X_train_scale = X_train/255.0
X_test_scale = X_test/255.0
clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(X_train_scale, Y_train)
# Predicting data accuracy
Y_predict = clf.predict(X_test_scale)
accuracy = accuracy_score(Y_test, Y_predict)
print("Accuracy of Logistic Regression:", accuracy)

cap = cv2.VideoCapture(0)
while (True):
    try:
        ret, frame = cap.read()
        # convert to greyscale image
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = grey.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right =(int(width/2 + 56), int(height/2 + 56))
        cv2.rectangle(grey, upper_left, bottom_right, (0,255,0), 2)
        roi = grey[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        im_pil= Image.fromarray(roi)
        # convert to landscape
        image_bw = im_pil.convert("L")
        image_bw_resize = image_bw.resize((28,28), Image.ANTIALIAS)
        # inverting image
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)
        # scaling image
        pixel_filter = 20
        min_pixels = np.percentile(image_bw_resize_inverted, pixel_filter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize_inverted-min_pixels, 0, 255)
        max_pixels = np.max(image_bw_resize_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixels
        # checks image and predicts
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)

        print("Predicted class is: ", test_pred)

        # Display the resulting frame
        cv2.imshow('frame',grey)
        if cv2.waitKey(1) & 0xFF == ord('esc'):
            break
        
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
