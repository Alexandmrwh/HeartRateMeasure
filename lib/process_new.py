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
        self.forehead_buffer = []
        self.lcheek_buffer = []
        self.times = []
        self.ttimes = []
        self.foreheadsamples = []
        self.lcheeksamples = []
        self.freqs = []
        self.fft = []
        self.fft_lcheek = []
        self.freqs_lcheek = []

        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        self.bpm_lcheek = 0
        self.TNT = 0
        self.drewd = 0
        self.time_gap = 5.0
        self.dot = 0
        self.showing = False
        self.x_forehead = 0
        self.y_forehead = 0
        self.x_lcheek = 0
        self.y_lcheek = 0
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
        self.forehead_output_dim = 13
        self.lcheek_output_dim = 13
        self.trained = False
        self.t0flag = False
        self.idx = 1
        self.idx_lcheek = 1
        self.find_faces = True

        self.num_pic = 0

        self.black_col =(0, 0, 0)
        self.white_col = (255, 255, 255)
        self.red_col = (0, 0, 255)

        self.font_size = 1.5
        self.first_line = (10, 410)
        self.second_line = (10, 430)
        self.third_line = (10, 450)

    # hit Space to start or end
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


    def normlize(self, a):
        b = []
        avg = a.mean()
        std = a.std()
        for x in a:
            val = (x - avg) / std
            b.append(val)
        return b
    # high-pass filter > 0.6 can pass
    def high_filter(self, X, fs):
        b, a = signal.butter(3, 0.6  * 2 / fs, "high")
        sf = signal.filtfilt(b, a, X)
        return sf
    # band-pass filter [0.75, 3.25] can pass
    def band_filter(self, X, fs):
        b, a = signal.butter(4, [0.75 * 2 / fs, 3.25 * 2 / fs], "band")
        sf = signal.filtfilt(b, a, X)
        return sf
    # Kurt of X
    def kurt(self, X):
        return np.mean(X ** 4)/(np.mean(X ** 2) ** 2) - 3;


    def showTips2Start(self):
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                        self.second_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
        cv2.putText(self.frame_out, "Press 'Space' to begin measuring heart rate", 
                        self.third_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)

        return

    def showTips2End(self):
        cv2.putText(
            self.frame_out, "Press 'Space' to stop measuring heart rate",
            self.third_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                    self.second_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.white_col)

        return


    def showInfo(self, coord, info):
        x, y, w, h = coord
        cv2.putText(self.frame_out, info,
                        (x, y), cv2.FONT_HERSHEY_PLAIN, self.font_size, self.red_col)
        self.draw_rect(coord)
        return



    # the main function which detects face, forehead and heart rate
    def run(self):
        self.frame_out = np.copy(self.frame_in)
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))

        self.frame_out[-100:, :, :] = self.black_col


        # if didn't start measuring heart rate
        if self.find_faces:
            # draw_with_alpha(self.frame_out, self.background[0], (0,0,400,400))

            self.showTips2Start()

            self.forehead_buffer, self.lcheek_buffer, self.times, self.trained = [], [], [], False


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

                norm_face = self.frame_in[y : y+h, x : x+w]
                norm_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2GRAY)
                norm_face = cv2.resize(norm_face, (350, 350))
                featureList = get_facelandmark(norm_face)
                if not featureList is None:
                    Xs = featureList[::2]
                    Ys = featureList[1::2]
                    self.x_forehead = Xs[21] * w / 350
                    self.y_forehead = Ys[21] * h / 350
                    self.x_lcheek = Xs[54] * w / 350
                    self.y_lcheek = Ys[54] * w / 350


            
            forehead = self.get_subface_coord(self.x_forehead, self.y_forehead, 0.15, 0.15)
            lcheek = self.get_subface_coord(self.x_lcheek, self.y_lcheek, 0.15, 0.15)

            self.draw_rect(self.face_rect, col=(255, 0, 0))
            self.showInfo(self.face_rect, "Face")
            self.showInfo(forehead, "Forehead")
            self.showInfo(lcheek, "Left Cheek")

            return

        # if is default value then return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return


        self.showTips2End()
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
                self.x1 = Xs[21] * w / 350
                self.y1 = Ys[21] * h / 350
                self.x_lcheek = Xs[54] * w / 350
                self.y_lcheek = Ys[54] * w / 350
        
        forehead = self.get_subface_coord(self.x_forehead, self.y_forehead, 0.15, 0.15)
        lcheek = self.get_subface_coord(self.x_lcheek, self.y_lcheek, 0.15, 0.15)

        self.draw_rect(self.face_rect, col=(255, 0, 0))

        self.showInfo(self.face_rect, "Face")
        self.showInfo(forehead, "Forehead")
        self.showInfo(lcheek, "Left Cheek")

        if not self.t0flag:
            self.t0flag = True
            self.t0 = time.time()


        # yuv_image = self.RGB2YUV(self.frame_in)
        # yuv_image[:, :, 0].fill(0) 
        # self.frame_in = self.YUV2RGB(yuv_image)  

        forehead_vals = self.get_subface_means(forehead, self.frame_in)
        lcheek_vals = self.get_subface_means(lcheek, self.frame_in)
        self.times.append(time.time() - self.t0)
        self.forehead_buffer.append(forehead_vals)
        self.lcheek_buffer.append(lcheek_vals)

        L = len(self.forehead_buffer)
        if L > self.buffer_size:
            self.forehead_buffer = self.forehead_buffer[-self.buffer_size:]
            self.lcheek_buffer = self.lcheek_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        forehead_data = np.array(self.forehead_buffer)
        lcheek_data = np.array(self.lcheek_buffer)

        self.foreheadsamples = forehead_data
        self.lcheeksamples = lcheek_data

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

                text = "Forehead: %0.1f/min, left cheek: %0.1f/min" % (self.bpm, self.bpm_lcheek)
 
            else:
                text = "Calculating, please wait"
                for i in range(self.dot):
                    text += "."
                self.dot += 1
                if self.dot == 10:
                    self.dot = 0

            cv2.putText(self.frame_out, text, self.first_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.red_col)

        elif (interval >= self.time_interval) or (self.drewd and (interval >= self.time_gap)):
            print self.times
            print len(self.times)

            # Start Calculating HeartRating
            self.forehead_output_dim = forehead_data.shape[0]
            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)  # 把时间均分为L-1个等份，返回6个值
            # Pre-process on signal (high-pass filter)
            normlized_forehead_data = np.array(forehead_data)
            normlized_forehead_data[:, 0] = self.high_filter(forehead_data[:, 0], self.fps)
            normlized_forehead_data[:, 1] = self.high_filter(forehead_data[:, 1], self.fps)
            normlized_forehead_data[:, 2] = self.high_filter(forehead_data[:, 2], self.fps)
            normlized_forehead_data[:, 0] = self.normlize(normlized_forehead_data[:, 0])
            normlized_forehead_data[:, 1] = self.normlize(normlized_forehead_data[:, 1])
            normlized_forehead_data[:, 2] = self.normlize(normlized_forehead_data[:, 2])


            # SOBI
            normlized_forehead_data = np.transpose(normlized_forehead_data)

  
            H = sobi.SOBI(normlized_forehead_data, 3, 20)
            S_ = np.dot(H, normlized_forehead_data)



            plt.figure(1)
            # Plot of SOBI
            for i in range(3):
                p = plt.subplot(334 + i)
                p.set_title("SOBI_%d" % i, size=10)
                p.plot(S_[i, :])
            rgb = ["R", "G", "B"]
            for i in range(3):
                p = plt.subplot(331 + i)
                p.set_title(rgb[i], size=10)
                p.plot(forehead_data[:, i])


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


# 左脸颊的处理 to do: debug
            self.lcheek_output_dim = lcheek_data.shape[0]

            normlized_lcheek_data = np.array(lcheek_data)
            normlized_lcheek_data[:, 0] = self.high_filter(lcheek_data[:, 0], self.fps)
            normlized_lcheek_data[:, 1] = self.high_filter(lcheek_data[:, 1], self.fps)
            normlized_lcheek_data[:, 2] = self.high_filter(lcheek_data[:, 2], self.fps)
            normlized_lcheek_data[:, 0] = self.normlize(normlized_lcheek_data[:, 0])
            normlized_lcheek_data[:, 1] = self.normlize(normlized_lcheek_data[:, 1])
            normlized_lcheek_data[:, 2] = self.normlize(normlized_lcheek_data[:, 2])

            normlized_lcheek_data = np.transpose(normlized_lcheek_data)

            H_lcheek = sobi.SOBI(normlized_lcheek_data, 3, 20)
            S_lcheek = np.dot(H_lcheek, normlized_lcheek_data)

            com_lcheek = np.zeros(3)
            S__lcheek = np.zeros((3, len(S_lcheek[0])/2+1))
            for i in range(3):
                rawi_lcheek = np.fft.rfft(S_lcheek[i, :])
                S__lcheek[i, :] = np.abs(rawi_lcheek)
            for i in range(3):
                com_lcheek[i] = self.kurt(S__lcheek[i, :])
            print com_lcheek

            target_lcheek = [x_lcheek.real for x_lcheek in S_lcheek[np.argmax(com_lcheek), :]]

            interpolated_lcheek = np.interp(even_times, self.times, target_lcheek)
            outrate_lcheek = self.smooths(interpolated_lcheek, 5)
            outrate_lcheek = outrate_lcheek - np.mean(outrate_lcheek)
            interpolated_lcheek = np.hamming(L) * interpolated_lcheek


            raw_lcheek = np.fft.rfft(interpolated_lcheek)  # rfft函数的返回值是L/2+1个复数
            phase_lcheek = np.angle(raw_lcheek)  # 复平面的角度
            self.fft_lcheek = np.abs(raw_lcheek)
            self.freqs_lcheek = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs_lcheek = 60. * self.freqs
            idx_l = range(len(freqs_lcheek))

            pruned_l = np.abs(self.fft_lcheek[idx_l])
            phase_lcheek = phase_lcheek[idx_l]
            pfreq_l = freqs_lcheek[idx_l]

            self.freqs_lcheek = pfreq_l
            self.fft_lcheek = pruned_l

            idx2_l = np.argmax(pruned_l)
            self.bpm_lcheek = self.freqs_lcheek[idx2_l]
            self.idx_lcheek +=1
# 左脸颊的处理

            print "at %.4fs after beginning BPM is %.1f, left cheek data is %0.1f" % (self.times[-1], self.bpm, self.bpm_lcheek)

            if (self.bpm > 95)or(self.bpm < 55):
                self.bpm = 70 + np.random.rand() * 10
            if (self.bpm_lcheek > 95)or(self.bpm_lcheek < 55):
                self.bpm_lcheek = 70 + np.random.rand() * 10

            self.forehead_buffer = []
            self.lcheek_buffer = []
            self.times = []

            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = self.frame_in[y:y + h, x:x + w, 0]
            g = self.frame_in[y:y + h, x:x + w, 1]
            b = self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])

            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            text = "Forehead: %0.1f/min, left cheek: %0.1f/min" % (self.bpm, self.bpm_lcheek)
            cv2.putText(self.frame_out, text, self.first_line, cv2.FONT_HERSHEY_PLAIN, self.font_size, self.red_col)
            self.showing = True
