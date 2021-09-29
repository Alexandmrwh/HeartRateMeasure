from imutils import face_utils
import dlib
import cv2
import glob
import numpy as np
from sklearn import svm, manifold, decomposition, discriminant_analysis
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt

shapePredictorPath = 'models/shape_predictor_68_face_landmarks.dat'
faceDetector = dlib.get_frontal_face_detector()
facialLandmarkPredictor = dlib.shape_predictor(shapePredictorPath)

def get_facelandmark(grayImage):
    global faceDetector, facialLandmarkPredictor
    # face = image_as_nparray(grayImage)
    face = faceDetector(grayImage, 1)
    if len(face) == 0:
        return None

    facialLandmarks = facialLandmarkPredictor(grayImage, face[0])
    facialLandmarks = face_utils.shape_to_np(facialLandmarks)

    (x31, y31) = facialLandmarks[30]
    xyList = []
    for (x, y) in facialLandmarks[0:]:
        # xyList.append((x - x31, y - y31))
        xyList.append(x)
        xyList.append(y)

    # normalize
    xyArray = np.array(xyList)
    mu = xyArray.mean()
    # sigma = xyArray.std()
    # xyArray = 100.0 * (xyArray - mu)/ sigma
    max = xyArray.max()
    min = xyArray.min()
    # print max, min
    # xyArray = 100.0 * (xyArray - min) / (max - min)
    # xyList = list(xyArray)

    return xyList

