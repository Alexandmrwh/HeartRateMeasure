#coding=utf-8

from lib.device import Camera
from lib.getHR_v2 import getHR
import cv2
import numpy as np
import datetime
import serial
import socket
import sys


class getHeartRate(object):

    def __init__(self):
        self.cameras = []
        self.selected_cam = 0
        for i in range(0, 4):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.h = 0, 0
        self.pressed = 0

        self.processor = getHR(bpm_limits=[50, 160],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

# start or end to show heart rate
    def toggle_search(self):
        state = self.processor.changeState()
        print "face detection lock =", not state

# control of keys
    def key_handler(self):

        self.pressed = cv2.waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print "Exiting"
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

        if self.pressed == 32:
            print "Start Detecting"
            self.toggle_search()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run()
        # collect the output frame for display
        output_frame = self.processor.frame_out
        time_line = self.processor.time_line
        
        # cv2.namedWindow("Heart Rate", 0)
        # cv2.resizeWindow("Heart Rate", 800, 500)

        # show the processed/annotated output frame
        cv2.imshow("Heart Rate", output_frame)
        cv2.imshow("Time Line", time_line)

        # handle any key presses
        self.key_handler()

if __name__ == "__main__":

    hr = getHeartRate()
    cv2.namedWindow("Heart Rate", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Time Line", cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty("Heart Rate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        hr.main_loop()
