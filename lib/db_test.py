# coding = utf-8

import numpy as np
import time
import cv2
import os
import sys
import sobi
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy import signal
from peakDetection import peakdet
import csv
from scipy.fftpack import fft, ifft

def smooths(a, lenX):
	smoothed = []
	for i in range(0, len(a)):
		avg = 0.
		for j in range(0, lenX):
			if i + j < len(a):
				avg += float(a[i + j])
			else:
				avg += float(a[-1])
		smoothed.append(avg / lenX)
	return smoothed

# Function: normlize an array
def normlize(a):
	b = []
	avg = a.mean()
	std = a.std()
	for x in a:
		val = (x - avg) / std
		b.append(val)
	return b

# Function: high-pass filter, >threshold can pass
def high_filter(X, threshold, fs):
	b, a = signal.butter(3, threshold * 2 / fs, "high")
	sf = signal.filtfilt(b, a, X)
	return sf

# Function: band-pass filter, t1< <t2 can pass
def band_filter(X, t1, t2, fs):
	b, a = signal.butter(3, [t1 * 2 / fs, t2 * 2 / fs], "band")
	sf = signal.filtfilt(b, a, X)
	return sf

# Funtion: calculate kurt of X
def kurt(X):
	return np.mean(X ** 4)/(np.mean(X ** 2) ** 2) - 3;

def preProcess(rawData):
	normlizedData = np.array(rawData)
	normlizedData[:, 0] = high_filter(rawData[:, 0], 0.4, 500)
	normlizedData[:, 1] = high_filter(rawData[:, 1], 0.4, 500)
	normlizedData[:, 2] = high_filter(rawData[:, 2], 0.4, 500)
	normlizedData[:, 0] = normlize(normlizedData[:, 0])
	normlizedData[:, 1] = normlize(normlizedData[:, 1])
	normlizedData[:, 2] = normlize(normlizedData[:, 2])
	return normlizedData

# SOBI
def separatebySOBI(raw):
	H = sobi.SOBI(raw, 3, 20)
	source = np.dot(H, raw)
	
	return source

# calculate kurt
def calKurt(source):
	kurt_res = np.zeros(3)
	source_frequence = np.zeros((3, len(source[0])/2+1))
	for i in range(3):
		s_freq = np.fft.rfft(source[i, :])
		source_frequence[i, :] = np.abs(s_freq)
	for i in range(3):
		kurt_res[i] = kurt(source_frequence[i, :])
	return kurt_res

def calBPMbyfft(ppg, tmpLen, fps):
	N = len(ppg)
	T = 1.0 / fps
	xf = np.linspace(0.0, 1.0 / (2.0*T), N/2)
	yf = fft(ppg)

	phase = np.abs(yf)

	return xf[np.argmax(phase[:N/2])] * 60.0

dataset_res = []

for i in range(1, 10):
	with open('../rrest-syn_csv/rrest-syn0%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		raw_str = [row for row in reader]

	raw_str = np.array(raw_str)
	raw = []
	raw = raw_str.astype(np.float)
	raw = raw.transpose()

	tmp_output_dimension = raw[0]
	
	if raw.shape[0] == 105000:
		tmplen = 105000
		tmp_fps = 500.0
		times = np.arange(0, 210, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 210, tmplen)
	elif raw.shape[0] == 104999:
		tmplen = 104999
		tmp_fps = 500.0
		times = np.arange(0, 209.998, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 209.998, tmplen)

	tmp_norm = preProcess(raw)
	tmp_norm = np.transpose(tmp_norm)
	tmp_sour = separatebySOBI(tmp_norm)
	tmp_kurt = calKurt(tmp_sour)
	tmp_target = [x.real for x in tmp_sour[np.argmax(tmp_kurt), :]]
	tmp_inter = np.interp(tmp_even_times, times, tmp_target)
	tmp_hr = smooths(tmp_inter, 5)
	tmp_hr = tmp_hr - np.mean(tmp_hr)
	tmp_inter = np.hamming(tmplen) * tmp_inter
	tmp_bpm = calBPMbyfft(tmp_inter, tmplen, tmp_fps)

	dataset_res.append(tmp_bpm)

print dataset_res

for i in range(10, 100):
	print i
	with open('../rrest-syn_csv/rrest-syn0%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		raw_str = [row for row in reader]

	raw_str = np.array(raw_str)
	raw = []
	raw = raw_str.astype(np.float)
	raw = raw.transpose()

	tmp_output_dimension = raw[0]
	
	if raw.shape[0] == 105000:
		tmplen = 105000
		tmp_fps = 500.0
		times = np.arange(0, 210, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 210, tmplen)
	elif raw.shape[0] == 104999:
		tmplen = 104999
		tmp_fps = 500.0
		times = np.arange(0, 209.998, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 209.998, tmplen)

	tmp_norm = preProcess(raw)
	tmp_norm = np.transpose(tmp_norm)
	tmp_sour = separatebySOBI(tmp_norm)
	tmp_kurt = calKurt(tmp_sour)
	tmp_target = [x.real for x in tmp_sour[np.argmax(tmp_kurt), :]]
	tmp_inter = np.interp(tmp_even_times, times, tmp_target)
	tmp_hr = smooths(tmp_inter, 5)
	tmp_hr = tmp_hr - np.mean(tmp_hr)
	tmp_inter = np.hamming(tmplen) * tmp_inter
	tmp_bpm = calBPMbyfft(tmp_inter, tmplen, tmp_fps)

	dataset_res.append(tmp_bpm)

print dataset_res


for i in range(100, 193):
	print i
	with open('../rrest-syn_csv/rrest-syn%d_data.csv' % i, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		raw_str = [row for row in reader]

	raw_str = np.array(raw_str)
	raw = []
	raw = raw_str.astype(np.float)
	raw = raw.transpose()

	tmp_output_dimension = raw[0]
	
	if raw.shape[0] == 105000:
		tmplen = 105000
		tmp_fps = 500.0
		times = np.arange(0, 210, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 210, tmplen)
	elif raw.shape[0] == 104999:
		tmplen = 104999
		tmp_fps = 500.0
		times = np.arange(0, 209.998, 0.002)
		times = np.array(times)
		tmp_even_times = np.linspace(0, 209.998, tmplen)
	tmp_norm = preProcess(raw)
	tmp_norm = np.transpose(tmp_norm)
	tmp_sour = separatebySOBI(tmp_norm)
	tmp_kurt = calKurt(tmp_sour)
	tmp_target = [x.real for x in tmp_sour[np.argmax(tmp_kurt), :]]
	tmp_inter = np.interp(tmp_even_times, times, tmp_target)
	tmp_hr = smooths(tmp_inter, 5)
	tmp_hr = tmp_hr - np.mean(tmp_hr)
	tmp_inter = np.hamming(tmplen) * tmp_inter
	tmp_bpm = calBPMbyfft(tmp_inter, tmplen, tmp_fps)

	dataset_res.append(tmp_bpm)



with open('../rrest-syn_csv/dataset_res.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(dataset_res)