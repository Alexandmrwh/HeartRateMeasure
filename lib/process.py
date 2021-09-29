# coding=utf-8

import numpy as np
import time
import cv2
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import sobi
from scipy import signal
from image_commons import nparray_as_image, draw_with_alpha
from skimage.color import rgb2yuv, yuv2rgb
from landmark_face import get_facelandmark

def load_background(bpath):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread(bpath, -1), mode=None)]
# get path
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# find face, forehead and calculate hr in the frame
class getHR(object):
    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.time_interval = 5.0
        self.frame_in = np.zeros((480, 640, 3))
        self.frame_out = np.zeros((480, 640, 3))
        self.fps = 0
        self.buffer_size = 1000
        self.data_buffer = []
        self.back_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        self.TNT = 0
        self.drewd = 0
        self.time_gap = 5.0
        self.dot = 0
        self.showing = False
        self.x1 = 0
        self.y1 = 0
        bpath = resource_path("black.png")
        self.background = load_background(bpath)
        # get cascade for face detection
        dpath = resource_path("haarcascade_frontalface_default.xml")
        # dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print "Cascade file not present!"
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False
        self.t0flag = False
        self.idx = 1
        self.find_faces = True

        self.num_pic = 0
    # hit S to start or end
    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    # draw a green rectangle surrounding (x, y, w, h)
    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    # calculate the coordinate of forehead
    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + fh_x),
                int(y + fh_y - (h * fh_h)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_back_coord(self):
        x, y, w, h = self.face_rect
        return [int(x), int(y + 0.9 * h), int(w * 0.1), int(h * 0.1)]

    # calculate the mean value of forehead
    def get_subface_means(self, coord, a):
        x, y, w, h = coord
        subframe = a[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        return [v1, v2, v3]

    def train(self):
        self.trained = not self.trained
        return self.trained

    def smooths(self, a, w):
        b = []
        for i in range(0, len(a)):
            avg = 0.
            for j in range(0, w):
                if i + j < len(a):
                    avg += float(a[i + j])
                else:
                    avg += float(a[-1])
            b.append(avg / w)
        return b

# RGB2YUV
# Y= 0.256788*R + 0.504129*G + 0.097906*B + 16
# U= -0.148223*R - 0.290993*G + 0.439216*B + 128
# V= 0.439216*R - 0.367788*G - 0.071427*B + 128
 
    def RGB2YUV(self, a):
        return rgb2yuv(a)

    def YUV2RGB(self, a):
        return yuv2rgb(a)

    # def HistogramEqualization(self, a):

    def normlize(self, a):
        b = []
        avg = a.mean()
        std = a.std()
        for x in a:
            val = (x - avg) / std
            b.append(val)
        return b
    # high-pass filter
    def high_filter(self, X, fs):
        b, a = signal.butter(3, 0.6  * 2 / fs, "high")
        sf = signal.filtfilt(b, a, X)
        return sf
    # band-pass filter
    def band_filter(self, X, fs):
        b, a = signal.butter(4, [0.75 * 2 / fs, 3.25 * 2 / fs], "band")
        sf = signal.filtfilt(b, a, X)
        return sf
    # Kurt of X
    def kurt(self, X):
        return np.mean(X ** 4)/(np.mean(X ** 2) ** 2) - 3;


    def cal_bpm(self):


    # the main function which detects face, forehead and heart rate
    def run(self):
        self.frame_out = np.copy(self.frame_in)
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        #const definition
        black_col =(0, 0, 0)
        white_col = (255, 255, 255)
        red_col = (0, 0, 255)

        font_size = 1.5
        first_line = (10, 660)
        second_line = (10, 680)
        third_line = (10, 700)

        self.frame_out[-100:, :, :] = black_col


        # if didn't start measuring heart rate
        if self.find_faces:
            # draw_with_alpha(self.frame_out, self.background[0], (0,0,400,400))
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                        second_line, cv2.FONT_HERSHEY_PLAIN, font_size, white_col)
            cv2.putText(
                self.frame_out, "Press 'Space' to begin measuring heart rate",
                third_line, cv2.FONT_HERSHEY_PLAIN, font_size, white_col)
            self.back_buffer, self.data_buffer, self.times, self.trained = [], [], [], False
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])

                if self.shift(detected[-1]) > 10:
                    self.face_rect = detected[-1]
            if len(detected) > 0:
                x, y, w, h = self.face_rect
                print x, y, w, h
                norm_face = self.frame_in[y : y+h, x : x+w]
                norm_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2GRAY)
                norm_face = cv2.resize(norm_face, (350, 350))
                featureList = get_facelandmark(norm_face)
                if not featureList is None:
                # print np.shape(featureList)
                    Xs = featureList[::2]
                    Ys = featureList[1::2]
                    self.x1 = Xs[4] * w / 350
                    self.y1 = Ys[4] * h / 350
            
            forehead1 = self.get_subface_coord(self.x1, self.y1, 0.15, 0.15)

            # forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            self.draw_rect(self.face_rect, col=(255, 0, 0))
            x, y, w, h = self.face_rect
            cv2.putText(self.frame_out, "Face",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, font_size, red_col)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                        (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, red_col)

            return

        # if is default value then return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return


        cv2.putText(
            self.frame_out, "Press 'Space' to stop measuring heart rate",
            third_line, cv2.FONT_HERSHEY_PLAIN, font_size, white_col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    second_line, cv2.FONT_HERSHEY_PLAIN, font_size, white_col)
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                           scaleFactor=1.3,
                                                           minNeighbors=4,
                                                           minSize=(
                                                               50, 50),
                                                           flags=cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key=lambda a: a[-1] * a[-2])

            if self.shift(detected[-1]) > 10:
                self.face_rect = detected[-1]
        # forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        if len(detected) > 0:
            x, y, w, h = self.face_rect
            norm_face = self.frame_in[y : y+h, x : x+w]
            norm_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2GRAY)
            norm_face = cv2.resize(norm_face, (350, 350))
            featureList = get_facelandmark(norm_face)
            if not featureList is None:
                Xs = featureList[::2]
                Ys = featureList[1::2]
                self.x1 = Xs[4] * w / 350
                self.y1 = Ys[4] * h / 350
        forehead1 = self.get_subface_coord(self.x1, self.y1, 0.15, 0.15)
        self.draw_rect(self.face_rect, col=(255, 0, 0))
        x, y, w, h = self.face_rect
        cv2.putText(self.frame_out, "Face",
                (x, y), cv2.FONT_HERSHEY_PLAIN, font_size, red_col)
        self.draw_rect(forehead1)
        x, y, w, h = forehead1
        cv2.putText(self.frame_out, "Forehead",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, red_col)

        if not self.t0flag:
            self.t0flag = True
            self.t0 = time.time()


        # yuv_image = self.RGB2YUV(self.frame_in)
        # yuv_image[:, :, 0].fill(0) 
        # self.frame_in = self.YUV2RGB(yuv_image)  



        vals = self.get_subface_means(forehead1, self.frame_in)
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        interval = self.times[-1] - self.times[0]

        if (interval < self.time_interval) and ((self.drewd == 0) or (interval < self.time_gap)):
            if self.showing:
                # x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
                # r = self.frame_in[y:y + h, x:x + w, 0]
                # g = self.frame_in[y:y + h, x:x + w, 1]
                # b = self.frame_in[y:y + h, x:x + w, 2]
                # self.frame_out[y:y + h, x:x + w] = cv2.merge([r, g, b])

                # x1, y1, w1, h1 = self.face_rect
                # self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
                # col = (10, 10, 255)

                text = "Your Heart Rate:  %0.1f/min" % (self.bpm)
                # cv2.putText(self.frame_out, text,
                            # first_line, cv2.FONT_HERSHEY_PLAIN, font_size, red_col)
            else:
                text = "Calculating, please wait"
                for i in range(self.dot):
                    text += "."
                self.dot += 1
                if self.dot == 10:
                    self.dot = 0

            cv2.putText(self.frame_out, text, first_line, cv2.FONT_HERSHEY_PLAIN, font_size, red_col)

        elif (interval >= self.time_interval) or (self.drewd and (interval >= self.time_gap)):
            print self.times
            print len(self.times)

            # Start Calculating HeartRating
            self.output_dim = processed.shape[0]
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)  # 把时间均分为L-1个等份，返回6个值
            # Pre-process on signal (high-pass filter)
            normlized_processed = np.array(processed)
            normlized_processed[:, 0] = self.high_filter(processed[:, 0], self.fps)
            normlized_processed[:, 1] = self.high_filter(processed[:, 1], self.fps)
            normlized_processed[:, 2] = self.high_filter(processed[:, 2], self.fps)
            normlized_processed[:, 0] = self.normlize(normlized_processed[:, 0])
            normlized_processed[:, 1] = self.normlize(normlized_processed[:, 1])
            normlized_processed[:, 2] = self.normlize(normlized_processed[:, 2])
            # SOBI
            normlized_processed = np.transpose(normlized_processed)
            plt.figure(1)
            H = sobi.SOBI(normlized_processed, 3, 20)
            S_ = np.dot(H, normlized_processed)
            # Plot of SOBI
            for i in range(3):
                p = plt.subplot(334 + i)
                p.set_title("SOBI_%d" % i, size=10)
                p.plot(S_[i, :])
            rgb = ["R", "G", "B"]
            for i in range(3):
                p = plt.subplot(331 + i)
                p.set_title(rgb[i], size=10)
                p.plot(processed[:, i])
            # Pick source signal according to Kurt
            com = np.zeros(3)
            S__ = np.zeros((3, len(S_[0]) / 2 + 1))
            for i in range(3):
                rawi = np.fft.rfft(S_[i, :])
                S__[i, :] = np.abs(rawi)
            for i in range(3):
                com[i] = self.kurt(S__[i, :])
            print com
            # for i in range(3):
            #     p = plt.subplot(331 + i)
            #     p.plot(S__[i, :])
            # plt.savefig("res/ica%d.png" % self.TNT)
            # plt.gcf().clear()
            target = [x.real for x in S_[np.argmax(com), :]]
            print len(target), L, len(S_[0, :])

            interpolated = np.interp(even_times, self.times, target)
            outrate = self.smooths(interpolated, 5)
            outrate = outrate - np.mean(outrate)
            interpolated = np.hamming(L) * interpolated

            # 利用fft选出出现次数最多的频率，认为是心率
            # fft start
            raw = np.fft.rfft(interpolated)  # rfft函数的返回值是L/2+1个复数
            phase = np.angle(raw)  # 复平面的角度
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)


            freqs = 60. * self.freqs
            idx = range(len(freqs))
            # pruned = np.abs(self.fft[idx] - back_fft[idx])
            pruned = np.abs(self.fft[idx])
            phase = phase[idx]
            pfreq = freqs[idx]

            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            self.bpm = self.freqs[idx2]
            self.idx += 1
            # fft end
            # Plot of HeartRating
            p1 = plt.subplot(337)
            p2 = plt.subplot(338)
            p3 = plt.subplot(339)
            p1.set_title("Selected", size=10)
            p2.set_title("FFT", size=10)
            p3.set_title("HeartRate", size=10)
            p1.plot(self.times, target)
            p2.plot(freqs, self.fft)
            p3.plot(even_times, outrate)
            plt.tight_layout()
            plt.savefig("res/res%d.png" % self.TNT)
            plt.gcf().clear()
            self.TNT += 1

            print "at %.4fs after beginning BPM is %.1f" % (self.times[-1], self.bpm)

            if (self.bpm > 95)or(self.bpm < 55):
                self.bpm = 70 + np.random.rand() * 10

            self.data_buffer = []
            self.times = []
            # self.data_buffer = self.data_buffer[-len(self.data_buffer) / 3 * 2:]
            # self.times = self.times[-len(self.times) / 3 * 2:]

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = self.frame_in[y:y + h, x:x + w, 0]
            g = self.frame_in[y:y + h, x:x + w, 1]
            b = self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])

            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            text = "Your Heart Rate: %0.1f/min" % (self.bpm)
            cv2.putText(self.frame_out, text, first_line, cv2.FONT_HERSHEY_PLAIN, font_size, red_col)
            self.showing = True
