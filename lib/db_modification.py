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

'''
add gaussian noise
'''

# for i in range(1, 10):
# 	with open('../rrest-syn_csv_raw/rrest-syn00%d_data.csv' % i, 'rb') as csvfile:
# 		reader = csv.reader(csvfile)
# 		ppg_str = [row[0] for row in reader]

# 		ppg = [float(x) for x in ppg_str]
# 		ppg = np.array(ppg)
# 		noise0 = np.random.normal(0, 1, ppg.shape)
# 		noise1 = np.random.normal(0, 1, ppg.shape)
# 		noise2 = np.random.normal(0, 1, ppg.shape)
# 		ppg_withnoise0 = ppg + noise0
# 		ppg_withnoise1 = ppg + noise1
# 		ppg_withnoise2 = ppg + noise2

# 	with open('../rrest-syn_csv/rrest-syn00%d_data.csv' % i, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter = ',')
# 		writer.writerow(ppg_withnoise0)
# 		writer.writerow(ppg_withnoise1)
# 		writer.writerow(ppg_withnoise2)


# for i in range(10, 100):
# 	with open('../rrest-syn_csv_raw/rrest-syn0%d_data.csv' % i, 'rb') as csvfile:
# 		reader = csv.reader(csvfile)
# 		ppg_str = [row[0] for row in reader]

# 		ppg = [float(x) for x in ppg_str]
# 		ppg = np.array(ppg)
# 		noise0 = np.random.normal(0, 1, ppg.shape)
# 		noise1 = np.random.normal(0, 1, ppg.shape)
# 		noise2 = np.random.normal(0, 1, ppg.shape)
# 		ppg_withnoise0 = ppg + noise0
# 		ppg_withnoise1 = ppg + noise1
# 		ppg_withnoise2 = ppg + noise2

# 	with open('../rrest-syn_csv/rrest-syn0%d_data.csv' % i, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter = ',')
# 		writer.writerow(ppg_withnoise0)
# 		writer.writerow(ppg_withnoise1)
# 		writer.writerow(ppg_withnoise2)

# for i in range(100, 193):
# 	with open('../rrest-syn_csv_raw/rrest-syn%d_data.csv' % i, 'rb') as csvfile:
# 		reader = csv.reader(csvfile)
# 		ppg_str = [row[0] for row in reader]

# 		ppg = [float(x) for x in ppg_str]
# 		ppg = np.array(ppg)
# 		noise0 = np.random.normal(0, 1, ppg.shape)
# 		noise1 = np.random.normal(0, 1, ppg.shape)
# 		noise2 = np.random.normal(0, 1, ppg.shape)
# 		ppg_withnoise0 = ppg + noise0
# 		ppg_withnoise1 = ppg + noise1
# 		ppg_withnoise2 = ppg + noise2

# 	with open('../rrest-syn_csv/rrest-syn%d_data.csv' % i, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter = ',')
# 		writer.writerow(ppg_withnoise0)
# 		writer.writerow(ppg_withnoise1)
# 		writer.writerow(ppg_withnoise2)

'''
add random noise
'''

for i in range(1, 2):
	with open('../rrest-syn_csv_raw/rrest-syn00%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		ppg_str = [row[0] for row in reader]

		ppg = [float(x) for x in ppg_str]
		ppg = np.array(ppg)

		print ppg.shape

		noise0 = np.random.uniform(low = -2.0, high = 2.0, size = (105000,))
		noise1 = np.random.uniform(low = -2.0, high = 2.0, size = (105000,))
		noise2 = np.random.uniform(low = -2.0, high = 2.0, size = (105000,))

		ppg_withnoise0 = ppg + noise0
		ppg_withnoise1 = ppg + noise1
		ppg_withnoise2 = ppg + noise2

		even_times = np.linspace(0, 10, 5000)

		plt.figure(1)
		plt.title("Generated rPPG Signal")
		plt.xlim((0, 10))
		plt.ylim((-4.0, 4.0))
		plt.xlabel("Sampling Points")
		plt.plot(even_times, ppg_withnoise0[:5000])
		plt.grid(True)
		plt.legend()
		plt.show()




	# with open('../rrest-syn_csv/rrest-syn00%d_data.csv' % i, 'wb') as csvfile:
	# 	writer = csv.writer(csvfile, delimiter = ',')
	# 	writer.writerow(ppg_withnoise0)
	# 	writer.writerow(ppg_withnoise1)
	# 	writer.writerow(ppg_withnoise2)


# for i in range(10, 100):
# 	with open('../rrest-syn_csv_raw/rrest-syn0%d_data.csv' % i, 'rb') as csvfile:
# 		reader = csv.reader(csvfile)
# 		ppg_str = [row[0] for row in reader]

# 		ppg = [float(x) for x in ppg_str]
# 		ppg = np.array(ppg)
# 		noise0 = np.random(ppg.shape)
# 		noise1 = np.random(ppg.shape)
# 		noise2 = np.random(ppg.shape)
# 		ppg_withnoise0 = ppg + noise0
# 		ppg_withnoise1 = ppg + noise1
# 		ppg_withnoise2 = ppg + noise2

# 	with open('../rrest-syn_csv/rrest-syn0%d_data.csv' % i, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter = ',')
# 		writer.writerow(ppg_withnoise0)
# 		writer.writerow(ppg_withnoise1)
# 		writer.writerow(ppg_withnoise2)

# for i in range(100, 193):
# 	with open('../rrest-syn_csv_raw/rrest-syn%d_data.csv' % i, 'rb') as csvfile:
# 		reader = csv.reader(csvfile)
# 		ppg_str = [row[0] for row in reader]

# 		ppg = [float(x) for x in ppg_str]
# 		ppg = np.array(ppg)
# 		noise0 = np.random(ppg.shape)
# 		noise1 = np.random(ppg.shape)
# 		noise2 = np.random(ppg.shape)
# 		ppg_withnoise0 = ppg + noise0
# 		ppg_withnoise1 = ppg + noise1
# 		ppg_withnoise2 = ppg + noise2

# 	with open('../rrest-syn_csv/rrest-syn%d_data.csv' % i, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter = ',')
# 		writer.writerow(ppg_withnoise0)
# 		writer.writerow(ppg_withnoise1)
# 		writer.writerow(ppg_withnoise2)

