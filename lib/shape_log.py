# coding = utf-8

import numpy as np
import time
import cv2
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy import signal
from peakDetection import peakdet
import csv

for i in range(1, 10):
	with open('../rrest-syn_csv_raw/rrest-syn00%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		ppg_str = [row[0] for row in reader]

		ppg = [float(x) for x in ppg_str]
		ppg = np.array(ppg)
		print i, ppg.shape

for i in range(10, 100):
	with open('../rrest-syn_csv_raw/rrest-syn0%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		ppg_str = [row[0] for row in reader]

		ppg = [float(x) for x in ppg_str]
		ppg = np.array(ppg)
		print i, ppg.shape

for i in range(100, 193):
	with open('../rrest-syn_csv_raw/rrest-syn%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		ppg_str = [row[0] for row in reader]

		ppg = [float(x) for x in ppg_str]
		ppg = np.array(ppg)
		print i, ppg.shape


