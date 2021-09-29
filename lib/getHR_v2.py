# coding = utf-8

import numpy as np
import time
import cv2
import os
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from sklearn.decomposition import FastICA
import sobi
from scipy import signal
from image_commons import nparray_as_image, draw_with_alpha
from landmark_face import get_facelandmark
from peakDetection import peakdet

# get path
def resource_path(relative_path):
	try:
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")
	return os.path.join(base_path, relative_path)

class getHR(object):

	# Function: initiate

	def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):
		self.frame_in = np.zeros((480, 640, 3))
		self.frame_out = np.zeros((480, 640, 3))
		self.grayframe = np.zeros((480, 640, 3))
		self.face_rect = [1, 1, 2, 2]
		cascade_path = resource_path("haarcascade_frontalface_default.xml")
		if not os.path.exists(cascade_path):
			print "Cascade file not present!"
		self.face_cascade = cv2.CascadeClassifier(cascade_path)

		self.forehead = [0, 0, 1, 1]
		self.foreheadx = 0
		self.foreheady = 0

		self.cheek0 = [0, 0, 1, 1]
		self.cheek0x = 0
		self.cheek0y = 0

		self.cheek1 = [0, 0, 1, 1]
		self.cheek1x = 0
		self.cheek1y = 0

		self.t0 = time.time()

		self.alltime = []
		self.all_fh = []
		self.all_c0 = []
		self.all_samples_fh = []
		self.all_source_fh = []
		self.all_target_fh = []
		self.all_samples_c0 = []
		self.all_source_c0 = []
		self.all_target_c0 = []

		self.even_times = []

		self.timestamp = []
		self.ROI_forehead = []
		self.ROI_cheek0 = []
		self.ROI_cheek1 = []
		self.fh_samples = []
		self.c0_samples = []

		self.source_fh = []
		self.source_c0 = []
		self.target_fh = []
		self.target_c0 = []
		self.HR_fh = []
		self.HR_c0 = []

		self.fps = 0.0
		self.ppg = []
		self.fh_bmp = 0
		self.c0_bpm = 0

		self.ist0 = False
		self.isCalculating = False
		self.hasResult = False

		self.picnum = 0

		self.last_center = np.array([0, 0])

		# const definition
		self.dot = 0
		self.mini_interval = 5.0
		self.bufferSize = 1024
		self.black_col = (0, 0, 0)
		self.white_col = (255, 255, 255)
		self.red_col = (0, 0, 255)
		self.green_col = (0, 255, 0)

		self.font_size = 1.5
		self.first_line = (10, 410)
		self.second_line = (10, 430)
		self.third_line = (10, 450)

		self.time_line = np.zeros((480, 640, 3))
		self.fh_x0 = 50
		self.fh_y0 = 300
		self.c0_x0 = 50
		self.c0_y0 = 300
		self.time_buf_x = []
		self.time_buf_fh = []
		self.time_buf_c0 = []
		self.time_buf_idx = 0
		self.time_buf_len = 0


	# Function: record calculating state

	def changeState(self):
		# all result
		if self.isCalculating:
			self.all_samples_fh = np.array(self.all_fh)
			self.all_samples_c0 =  np.array(self.all_c0)

			tmp_output_dimension = self.all_samples_fh.shape[0]
			tmplen = len(self.all_fh)
			tmp_fps = tmplen / (self.alltime[-1] - self.alltime[0])
			tmp_even_times = np.linspace(self.alltime[0], self.alltime[-1], tmplen)

			tmp_norm_fh = self.preProcess(self.all_samples_fh)
			tmp_norm_fh = np.transpose(tmp_norm_fh)
			tmp_sour_fh = self.separatebySOBI(tmp_norm_fh)
			tmp_kurt_fh = self.calKurt(tmp_sour_fh)
			tmp_target_fh = [x.real for x in tmp_sour_fh[np.argmax(tmp_kurt_fh), :]]
			tmp_inter_fh = np.interp(tmp_even_times, self.alltime, tmp_target_fh)
			tmp_hr_fh = self.smooths(tmp_inter_fh, 5)
			tmp_hr_fh = tmp_hr_fh - np.mean(tmp_hr_fh)
			tmp_inter_fh = np.hamming(tmplen) * tmp_inter_fh
			tmp_fh_bpm = self.calBPM(tmp_inter_fh, tmplen, tmp_fps)

			tmp_norm_c0 = self.preProcess(self.all_samples_c0)
			tmp_norm_c0 = np.transpose(tmp_norm_c0)
			tmp_sour_c0 = self.separatebySOBI(tmp_norm_c0)
			tmp_kurt_c0 = self.calKurt(tmp_sour_c0)
			tmp_target_c0 = [x.real for x in tmp_sour_c0[np.argmax(tmp_kurt_c0), :]]
			tmp_inter_c0 = np.interp(tmp_even_times, self.alltime, tmp_target_c0)
			tmp_hr_c0 = self.smooths(tmp_inter_c0, 5)
			tmp_hr_c0 = tmp_hr_c0 - np.mean(tmp_hr_c0)
			tmp_inter_c0 = np.hamming(tmplen) * tmp_inter_c0
			tmp_c0_bpm = self.calBPM(tmp_inter_c0, tmplen, tmp_fps)

			# peaks, valleys, peaktab, valleytab = peakdet(tmp_inter_c0, 0.1)
			# avg_bpm = peaks / (self.alltime[-1] - self.alltime[0]) * 60
			# plt.plot(tmp_inter_c0)
			# plt.scatter(np.array(peaktab)[:,0], np.array(peaktab)[:,1], color='blue')
			# plt.scatter(np.array(valleytab)[:,0], np.array(valleytab)[:,1], color='red')
			# plt.savefig("res/allpeaks_%d" % self.picnum)
			# plt.gcf().clear()

			plt.figure(1)
			plt.title("Observed Signal of Forehead")
			rgb = ['R', 'G', 'B']
			for i in range(3):
				color = plt.subplot(131+i)
				color.set_title(rgb[i] + "Chanel")
				color.set_xlim(((self.alltime[0], self.alltime[-1])))
				color.set_ylim((-2.0, 2.0))
				color.plot(tmp_even_times, self.all_samples_fh[:, i])
				color.grid(True)
				color.legend()
			
			plt.savefig("res/16s_fh_observed.png")
			plt.gcf().clear()

			plt.figure(1)
			plt.title("Heartbeat Curve with Head Motions")
			# plt.title("Heartbeat Curve During Conversation")
			plt.xlim((self.alltime[0], self.alltime[-1]))
			plt.ylim((-2.0, 2.0))
			plt.xlabel("Time")
			plt.plot(tmp_even_times, tmp_hr_fh, color = "blue")
			# plt.plot(tmp_even_times, tmp_hr_c0, color = "red")
			plt.grid(True)
			plt.legend()
			plt.savefig("res/16s_fh.png")
			plt.gcf().clear()

			plt.figure(1)

			plt.title("Heartbeat Curve with Head Motions")
			plt.title("Heartbeat Curve of Cheek Data")
			# plt.title("Heartbeat Curve During Conversation")

			plt.xlim((self.alltime[0], self.alltime[-1]))
			plt.ylim((-2.0, 2.0))
			plt.xlabel("Time")
			# plt.plot(tmp_even_times, tmp_hr_fh, color = "blue")
			plt.plot(tmp_even_times, tmp_hr_c0, color = "red")
			plt.grid(True)
			plt.legend()
			plt.savefig("res/16s_c0.png")
			plt.gcf().clear()

			plt.figure(1)
			plt.title("Heartbeat Curve with Head Motions")
			# plt.title("Heartbeat Curve During Conversation")
			plt.title("Heartbeat Curve with Head Motions")
			plt.xlim((self.alltime[0], self.alltime[-1]))
			plt.ylim((-2.0, 2.0))
			plt.xlabel("Time")
			plt.plot(tmp_even_times, tmp_hr_fh, color = "blue", label = "Forehead Data")
			plt.plot(tmp_even_times, tmp_hr_c0, color = "red", label = "Cheek Data")
			plt.grid(True)
			plt.legend()
			plt.savefig("res/16s_c0_fh.png")
			plt.gcf().clear()



			print "all result: fft %f, %f" % (tmp_fh_bpm, tmp_c0_bpm)




			plt.figure(1)
			rgb = ["R", "G", "B"]
			roi = "allfh"
			for i in range(3):
				p = plt.subplot(331+i)
				p.set_title(roi + "_" + rgb[i], size = 10)
				p = plt.subplot(334+i)
				p.set_title(roi + "_SOBI_%d" % i, size = 10)
			p7 = plt.subplot(337)
			p8 = plt.subplot(338)
			p9 = plt.subplot(339)
			p7.set_title(roi + "_" + "Raw PPG", size = 10)
			p8.set_title(roi + "_" + "FFT", size = 10)
			p9.set_title(roi + "_" + "PPG", size = 10)

			for i in range(3):
				p = plt.subplot(331+i)
				p.plot(self.all_samples_fh[:, i])
			for i in range(3):
				p = plt.subplot(334+i)
				p.plot(tmp_sour_fh[i, :])
			p7.plot(self.alltime, tmp_target_fh)

			plt.tight_layout()
			plt.savefig("res/res_all_result_fh.png")
			plt.gcf().clear()
			
			plt.figure(1)
			roi = "allc0"
			for i in range(3):
				p = plt.subplot(331+i)
				p.set_title(roi + "_" + rgb[i], size = 10)
				p = plt.subplot(334+i)
				p.set_title(roi + "_SOBI_%d" % i, size = 10)
			p7 = plt.subplot(337)
			p8 = plt.subplot(338)
			p9 = plt.subplot(339)
			p7.set_title(roi + "_" + "Raw PPG", size = 10)
			p8.set_title(roi + "_" + "FFT", size = 10)
			p9.set_title(roi + "_" + "PPG", size = 10)

			for i in range(3):
				p = plt.subplot(331+i)
				p.plot(self.all_samples_c0[:, i])
			for i in range(3):
				p = plt.subplot(334+i)
				p.plot(tmp_sour_c0[i, :])
			p7.plot(self.alltime, tmp_target_c0)

			plt.tight_layout()
			plt.savefig("res/res_all_result_c0.png")
			plt.gcf().clear()
		# /all result

		self.isCalculating = not self.isCalculating
		return self.isCalculating

	# Function: update self.last_center
	def shift(self, detected):
		x, y, w, h = detected
		center = np.array([x + 0.5*w, y + 0.5*h])
		shift = np.linalg.norm(center - self.last_center)

		self.last_center = center
		return shift

	# Function: draw a rectangle surrounding a given coordinate
	def draw_rect(self, rect, col):
		x, y, w, h = rect
		cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

	# Function: calculate the coordinate of forehead
	def get_subface_coord(self, sf_x, sf_y, sf_w, sf_h):
		x, y, w, h = self.face_rect
		return [int(x + sf_x), \
				int(y + sf_y - (h * sf_h)), \
				int(w * sf_w), \
				int(h * sf_h)]

	# Function: calculate the mean value of a given rectangle
	def get_rect_mean(self, coord, frameA):
		x, y, w, h = coord
		subframeA = frameA[y:y+h, x:x+w, :]
		v1 = np.mean(subframeA[:, :, 0])
		v2 = np.mean(subframeA[:, :, 1])
		v3 = np.mean(subframeA[:, :, 2])
		return [v1, v2, v3]

	# Function: smooth the curve by a window(len = lenX)
	def smooths(self, a, lenX):
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
	def normlize(self, a):
		b = []
		avg = a.mean()
		std = a.std()
		for x in a:
			val = (x - avg) / std
			b.append(val)
		return b

	# Function: high-pass filter, >threshold can pass
	def high_filter(self, X, threshold, fs):
		b, a = signal.butter(3, threshold * 2 / fs, "high")
		sf = signal.filtfilt(b, a, X)
		return sf

	# Function: band-pass filter, t1< <t2 can pass
	def band_filter(self, X, t1, t2, fs):
		b, a = signal.butter(3, [t1 * 2 / fs, t2 * 2 / fs], "band")
		sf = signal.filtfilt(b, a, X)
		return sf

	# Funtion: calculate kurt of X
	def kurt(self, X):
		return np.mean(X ** 4)/(np.mean(X ** 2) ** 2) - 3;



	# Function: display info
	def showTips2Start(self):
		cv2.putText(self.frame_out, "Press 'Esc' to quit", \
                        self.second_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
		cv2.putText(self.frame_out, "Press 'Space' to begin measuring heart rate", \
                        self.third_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
	
	def showTips2End(self):
		cv2.putText(self.frame_out, "Press 'Esc' to quit", \
        	self.second_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
		cv2.putText(
            self.frame_out, "Press 'Space' to stop measuring heart rate", \
            self.third_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
	
	def showInfo(self, coord, info):
		x, y, w, h = coord
		cv2.putText(self.frame_out, info, \
        	(x, y), cv2.FONT_HERSHEY_PLAIN, self.font_size, self.red_col)
		self.draw_rect(coord, self.green_col)

	# Function: get roi coordinate and show roi from a given frame
	def getROI(self, raw_frame):
		detected = list(self.face_cascade.detectMultiScale(raw_frame, \
                                                           scaleFactor=1.3, \
                                                           minNeighbors=4, \
                                                           minSize=(50, 50), \
                                                           flags=cv2.CASCADE_SCALE_IMAGE))
		if len(detected) > 0:
			detected.sort(key = lambda a: a[-1] * a[-2])

			if self.shift(detected[-1]) > 10:
				self.face_rect = detected[-1]

		if len(detected) > 0:
			x, y, w, h = self.face_rect

			norm_face = self.grayframe[y: y+h, x: x+w]
			norm_face = cv2.resize(norm_face, (350, 350))
			featureList = get_facelandmark(norm_face)
			if not featureList is None:
				Xs = featureList[::2]
				Ys = featureList[1::2]
				self.foreheadx = Xs[21] * w / 350
				self.foreheady = Ys[21] * h / 350
				self.cheek0x = Xs[54] * w / 350
				self.cheek0y = Ys[54] * w / 350
		self.forehead = self.get_subface_coord(self.foreheadx, self.foreheady, 0.15, 0.15)
		self.cheek0 = self.get_subface_coord(self.cheek0x, self.cheek0y, 0.15, 0.15)

		self.showInfo(self.face_rect, "Face")
		self.showInfo(self.forehead, "Forehead")
		self.showInfo(self.cheek0, "Cheek0")

	# Function: pre-process on a signal
	def preProcess(self, rawData):
		normlizedData = np.array(rawData)
		normlizedData[:, 0] = self.high_filter(rawData[:, 0], 0.6, self.fps)
		normlizedData[:, 1] = self.high_filter(rawData[:, 1], 0.6, self.fps)
		normlizedData[:, 2] = self.high_filter(rawData[:, 2], 0.6, self.fps)
		normlizedData[:, 0] = self.normlize(normlizedData[:, 0])
		normlizedData[:, 1] = self.normlize(normlizedData[:, 1])
		normlizedData[:, 2] = self.normlize(normlizedData[:, 2])
		return normlizedData

	# SOBI
	def separatebySOBI(self, raw):
		H = sobi.SOBI(raw, 3, 20)
		source = np.dot(H, raw)
		
		return source

	# calculate kurt
	def calKurt(self, source):
		kurt_res = np.zeros(3)
		source_frequence = np.zeros((3, len(source[0])/2+1))
		for i in range(3):
			s_freq = np.fft.rfft(source[i, :])
			source_frequence[i, :] = np.abs(s_freq)
		for i in range(3):
			kurt_res[i] = self.kurt(source_frequence[i, :])
		return kurt_res

	# FFT calculation
	def calBPM(self, ppg, tmpLen, fps):
		raw_freq = np.fft.rfft(ppg)
		phase = np.angle(raw_freq)
		fft = np.abs(raw_freq)

		freq0 = float(fps) / tmpLen * np.arange(tmpLen / 2 + 1)
		freqs = 60. * freq0

		idx = range(len(freqs))
		pruned = np.abs(fft[idx])
		phase = phase[idx]
		pfreq = freqs[idx]

		freq = pfreq
		fft = pruned
		idx2 = np.argmax(pruned)

		return freq[idx2]

	# another fft calculation
	def calBPMbyfft(self, ppg, tmpLen, fps):
		N = len(ppg)
		T = 1.0 / fps
		xf = np.linspace(0.0, 1.0 / (2.0*T), N/2)
		yf = fft(ppg)

		# print len(yf), len(xf), N, tmpLen
		phase = np.abs(yf)
		# plt.figure(1)
		# plt.plot(xf, phase[:N/2])
		# plt.show()
		# return xf[np.argmax(phase[:N/2])] * 60.0

		phaseMax = np.argmax(phase[:N/2])
		freqMax = xf[phaseMax]
		

		return freqMax * 60.0
		


	# draw result pic
	def drawResult(self, roi, number):
		plt.figure(1)
		rgb = ["R", "G", "B"]
		for i in range(3):
			p = plt.subplot(331+i)
			p.set_title(roi + "_" + rgb[i], size = 10)
			p = plt.subplot(334+i)
			p.set_title(roi + "_SOBI_%d" % i, size = 10)
		p7 = plt.subplot(337)
		p8 = plt.subplot(338)
		p9 = plt.subplot(339)
		p7.set_title(roi + "_" + "Raw PPG", size = 10)
		p8.set_title(roi + "_" + "FFT", size = 10)
		p9.set_title(roi + "_" + "PPG", size = 10)

		if roi == "Forehead":
			for i in range(3):
				p = plt.subplot(331+i)
				p.plot(self.fh_samples[:, i])
			for i in range(3):
				p = plt.subplot(334+i)
				p.plot(self.source_fh[i, :])
			p7.plot(self.timestamp, self.target_fh)
			# p8.plot()
			p9.plot(self.even_times, self.HR_fh)
			plt.tight_layout()
			plt.savefig("res/res_%s_%d.png" % (roi, number))
			plt.gcf().clear()
		if roi == "Cheek0":
			for i in range(3):
				p = plt.subplot(331+i)
				p.plot(self.c0_samples[:, i])
			for i in range(3):
				p = plt.subplot(334+i)
				p.plot(self.source_c0[i, :])
			p7.plot(self.timestamp, self.target_c0)
			# p8.plot()
			p9.plot(self.even_times, self.HR_c0)
			plt.tight_layout()
			plt.savefig("res/res_%s_%d.png" % (roi, number))
			plt.gcf().clear()

	def draw_line(self):
		dx = self.time_buf_x[self.time_buf_idx]
		dy0 = -int(self.time_buf_fh[self.time_buf_idx] * 200) + 300
		# dy1 = -int(self.time_buf_c0[self.time_buf_idx] * 200) + 300
		self.time_buf_idx += 1
		x = int(dx*30) + 50
		# print x, dy0, dy1
		cv2.line(self.time_line, (self.fh_x0, self.fh_y0), (x, dy0), (0, 255, 0), 2)
		# cv2.line(self.time_line, (self.c0_x0, self.c0_y0), (x, dy1), (0, 0, 255), 2)
		self.fh_x0 = x
		self.fh_y0 = dy0
		# self.c0_x0 = x
		# self.c0_y0 = dy1

	def run(self):
		self.grayframe = cv2.equalizeHist(cv2.cvtColor(self.frame_in, \
			cv2.COLOR_BGR2GRAY))
		self.frame_out = np.copy(self.frame_in)
		self.frame_out[-100:, :, :] = self.black_col

		# if not measuring heart rate
		# clear buffer
		# show roi
		if not self.isCalculating:
			self.showTips2Start()
			self.ROI_forehead, self.ROI_cheek0, self.ROI_cheek1, self.timestamp = [], [], [], []
			self.alltime, self.all_c0,  self.all_fh = [], [], []
			self.getROI(self.grayframe)
			self.time_line = np.zeros((480, 640, 3))
			self.fh_x0 = 50
			self.fh_y0 = 300
			self.c0_x0 = 50
			self.c0_y0 = 300
			self.time_buf_x = []
			self.time_buf_fh = []
			self.time_buf_c0 = []
			self.time_buf_idx = 0
			self.time_buf_len = 0

		
		if set(self.face_rect) == set([1, 1, 2, 2]):
			return
		# if is measuring heart rate
		# show roi
		# collect roi data
		# calculate ppg
		# calculate heart rate
		if self.isCalculating:
			self.showTips2End()
			self.getROI(self.grayframe)

			# record starting time
			if not self.ist0:
				self.ist0 = True
				self.t0 = time.time()

			if self.time_buf_idx < self.time_buf_len:
				self.draw_line()

			fh_val = self.get_rect_mean(self.forehead, self.frame_in)
			c0_val = self.get_rect_mean(self.cheek0, self.frame_in)

			# collect roi data
			self.timestamp.append(time.time() - self.t0)
			self.ROI_forehead.append(fh_val)
			self.ROI_cheek0.append(c0_val)

			self.alltime.append(time.time() - self.t0)
			self.all_fh.append(fh_val)
			self.all_c0.append(c0_val)

			# ROI buffers follow FIFO
			tmpLen = len(self.ROI_forehead)
			if tmpLen > self.bufferSize:

				self.ROI_forehead = self.ROI_forehead[-self.bufferSize: ]
				self.ROI_cheek0 = self.ROI_cheek0[-self.bufferSize: ]
				self.timestamp = self.timestamp[-self.bufferSize: ]
				tmpLen = self.bufferSize


			interval = self.timestamp[-1] - self.timestamp[0]

			# if less than 5s(self.mini_interval), then keep collecting
			if interval < self.mini_interval:
				if self.hasResult:
					text = "Heart Rate: %0.1f/min" % self.fh_bpm
					# text = "Forehead: %0.1f/min, cheek0: %0.1f/min" % (self.fh_bpm, self.c0_bpm)
				else:
					text = "Calculating, please wait"
					for i in range(self.dot):
						text += "."
					self.dot += 1
					if self.dot == 10:
						self.dot = 0

				cv2.putText(self.frame_out, text, self.first_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.red_col)

			# if has collected enough data, then calculate
			elif interval >= self.mini_interval:
				# convert to ndarray
				self.fh_samples = np.array(self.ROI_forehead)
				self.c0_samples = np.array(self.ROI_cheek0)

				# calculate dimension and x axis
				self.output_dimension = self.fh_samples.shape[0]
				self.fps = float(tmpLen) / (self.timestamp[-1] - self.timestamp[0])
				self.even_times = np.linspace(self.timestamp[0], self.timestamp[-1], tmpLen)

				# pre-process on signal
				normlized_fh = self.preProcess(self.fh_samples)
				normlized_c0 = self.preProcess(self.c0_samples)

				# SOBI
				normlized_fh = np.transpose(normlized_fh)
				normlized_c0 = np.transpose(normlized_c0)

				self.source_fh = self.separatebySOBI(normlized_fh)
				self.source_c0 = self.separatebySOBI(normlized_c0)

				# pick source signal according to kurt
				kurt_res_fh = self.calKurt(self.source_fh)
				kurt_res_c0 = self.calKurt(self.source_c0)

				self.target_fh = [x.real for x in self.source_fh[np.argmax(kurt_res_fh), :]]
				self.target_c0 = [x.real for x in self.source_c0[np.argmax(kurt_res_c0), :]]

				# get hr curve and smooth it 
				interpolated_fh = np.interp(self.even_times, self.timestamp, self.target_fh)
				self.HR_fh = self.smooths(interpolated_fh, 5)
				self.HR_fh = self.HR_fh - np.mean(self.HR_fh)

				interpolated_c0 = np.interp(self.even_times, self.timestamp, self.target_c0)
				self.HR_c0 = self.smooths(interpolated_c0, 5)
				self.HR_c0 = self.HR_c0 - np.mean(self.HR_c0)

				self.time_buf_x = []
				self.time_buf_fh = []
				self.time_buf_c0 = []

				for i in range(tmpLen):
					self.time_buf_x.append(self.even_times[i])
					self.time_buf_fh.append(self.HR_fh[i])
					self.time_buf_c0.append(self.HR_c0[i])

				# self.time_buf_x = self.even_times
				# self.time_buf_fh = self.HR_fh 
				# self.time_buf_c0 = self.HR_c0
				self.time_buf_idx = 0
				self.time_buf_len = tmpLen


				# calculate bpm by fft
				interpolated_fh = np.hamming(tmpLen) * interpolated_fh
				self.fh_bpm = self.calBPMbyfft(interpolated_fh, tmpLen, self.fps)

				interpolated_c0 = np.hamming(tmpLen) * interpolated_c0
				self.c0_bpm = self.calBPMbyfft(interpolated_c0, tmpLen, self.fps)

				smo_curve = np.hamming(tmpLen) * self.HR_fh
				smo_bpm = self.calBPMbyfft(smo_curve, tmpLen, self.fps)


				print "tmp fft result: %f, %f, smoothed fft: %f" % (self.fh_bpm, self.c0_bpm, smo_bpm)


				# debug without kurt
				for i in range(3):
					tmpinterfh = np.interp(self.even_times, self.timestamp, self.source_fh[i, :])
					tmphr = self.smooths(tmpinterfh, 5)
					tmphr = tmphr - np.mean(tmphr)
					tmpinterfh = np.hamming(tmpLen) * tmpinterfh
					tmpbpmfft = self.calBPMbyfft(tmpinterfh, tmpLen, self.fps)

					tmppeaks, tmpvalleys, tmppeaktab, tmpvalleytab = peakdet(tmphr, 0.1)
					tmpbpm = tmppeaks / (self.timestamp[-1] - self.timestamp[0]) * 60
					print "without kurt using forehead %d: fft %f peak cal %f" % (i, tmpbpmfft, tmpbpm)
				# /debug

				# count pic num
				# print interpolated_c0
				peaks, valleys, peaktab, valleytab = peakdet(self.HR_c0, 0.1)
				avg_bpm = peaks / (self.timestamp[-1] - self.timestamp[0]) * 60
				print "c0 peaks: %d, avg_bpm: %f" % (peaks, avg_bpm)
				plt.plot(self.HR_c0)
				plt.scatter(np.array(peaktab)[:,0], np.array(peaktab)[:,1], color='blue')
				plt.scatter(np.array(valleytab)[:,0], np.array(valleytab)[:,1], color='red')
				plt.savefig("res/peaks_%d" % self.picnum)


				# pic
				self.drawResult("Forehead", self.picnum)
				self.drawResult("Cheek0", self.picnum)
				self.picnum += 1


				self.ROI_forehead = []
				self.ROI_cheek0 = []
				self.timestamp = []

				self.hasResult = True


































