"""surveystick - software for SurveyStick digital surveillance tool."""

import json
import os.path as path
import cv2 as cv
import numpy as np


class App:
    """Even the main program gets a docstring."""
    def __init__(self):
        self.constraints = [{'is_on': False, 'h_min': 179, 'h_max': 179, 's_min': 255, 's_max': 255, 'v_min': 255, 'v_max': 255} for i in range(0, 10)]
        self.colors = [None for i in range(0, 10)]
        self.masks = [None for i in range(0, 10)]
        self.edges = [None for i in range(0, 10)]
        self.grays = [None for i in range(0, 10)]
        self.contours = [None for i in range(0, 10)]
        self.centroids = [None for i in range(0, 10)]
        self.centers = [None for i in range(0, 10)]

        self.calibrating = False
        self.calibrating_color = 0
        self.refresh_calibration = False
        self.swatchn = np.zeros((256, 256, 3), np.uint8)
        self.swatchx = np.zeros((256, 256, 3), np.uint8)
        self.hsv_img = None
        self.cal_img = None
        self.proc_img = None


    def track_bar_changed(self, _):
        self.refresh_calibration = True
        self.saveConstraints()


    def switch_flipped(self, _):
        con = self.constraints[self.calibrating_color]
        con['is_on'] = not con['is_on']


    is_on_trackbar_name = '0 : off \n1 : on'

    def make_calibrating_ui(self):
        """Calibrate the HSV isolation parameters for color color_name."""
        # make the ui if we're coming from no ui
        window_name = "cal"
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        hor = np.hstack((self.swatchn, self.swatchx))
        cv.imshow("cal", hor)
        con = self.constraints[self.calibrating_color]
        cv.createTrackbar(App.is_on_trackbar_name, window_name, 1 if con['is_on'] else 0, 1, self.switch_flipped)
        cv.createTrackbar("Hue min", window_name, con['h_min'], 179, self.track_bar_changed)
        cv.createTrackbar("Hue max", window_name, con['h_max'], 179, self.track_bar_changed)
        cv.createTrackbar("Sat min", window_name, con['s_min'], 255, self.track_bar_changed)
        cv.createTrackbar("Sat max", window_name, con['s_max'], 255, self.track_bar_changed)
        cv.createTrackbar("Val min", window_name, con['v_min'], 255, self.track_bar_changed)
        cv.createTrackbar("Val max", window_name, con['v_max'], 255, self.track_bar_changed)


    def destroy_calibrating_ui(self):
        cv.destroyWindow("cal")
        cv.destroyWindow("mask")


    def set_ui_from_constraints(self):
        window_name = "cal"
        cal = self.constraints[self.calibrating_color]
        cv.setTrackbarPos(App.is_on_trackbar_name, window_name, 1 if cal['is_on'] else 0)
        cv.setTrackbarPos("Hue min", window_name, cal['h_min'])
        cv.setTrackbarPos("Hue max", window_name, cal['h_max'])
        cv.setTrackbarPos("Sat min", window_name, cal['s_min'])
        cv.setTrackbarPos("Sat max", window_name, cal['s_max'])
        cv.setTrackbarPos("Val min", window_name, cal['v_min'])
        cv.setTrackbarPos("Val max", window_name, cal['v_max'])


    def update_constraints_from_ui(self):
        window_name = "cal"
        cal = self.constraints[self.calibrating_color]
        cal['is_on'] = cv.getTrackbarPos(App.is_on_trackbar_name, window_name) == 1
        cal['h_min'] = cv.getTrackbarPos("Hue min", window_name)
        cal['h_max'] = cv.getTrackbarPos("Hue max", window_name)
        cal['s_min'] = cv.getTrackbarPos("Sat min", window_name)
        cal['s_max'] = cv.getTrackbarPos("Sat max", window_name)
        cal['v_min'] = cv.getTrackbarPos("Val min", window_name)
        cal['v_max'] = cv.getTrackbarPos("Val max", window_name)
        self.computeColorsFromTrackbars()


    def refresh_calibrationImage(self):
        """updates the gradient images in the calibration UI"""
        cal = self.constraints[self.calibrating_color]
        hn = cal['h_min']
        hx = cal['h_max']
        sn = cal['s_min']
        sx = cal['s_max']
        vn = cal['v_min']
        vx = cal['v_max']

        def fill(h, img):
            for y in range(0, 256):
                v = (vx - vn) * y / 256 + vn
                for x in range(0, 256):
                    s = (sx - sn) * x / 256 + sn

                    img[y, x, 0] = h
                    img[y, x, 1] = s
                    img[y, x, 2] = v

        fill(hn, self.swatchn)
        fill(hx, self.swatchx)
        self.swatchn = cv.cvtColor(self.swatchn, cv.COLOR_HSV2BGR)
        self.swatchx = cv.cvtColor(self.swatchx, cv.COLOR_HSV2BGR)
        hor = np.hstack((self.swatchn, self.swatchx))
        cv.imshow("cal", hor)


    def computeColorsFromTrackbars(self):
        for idx, con in enumerate(self.constraints):
            self.colors[idx] = (int((con['h_max'] + con['h_min']) / 2) % 255,
                                int((con['s_max'] + con['s_min']) / 2),
                                int((con['v_max'] + con['v_min']) / 2))


    def make_mask(self, img, color_idx):
        con = self.constraints[color_idx]
        if con['h_min'] > con['h_max']:
            mask0 = cv.inRange(img, 
                np.array([0,            con['s_min'], con['v_min']]), 
                np.array([con['h_max'], con['s_max'], con['v_max']]))
            mask1 = cv.inRange(img, 
                np.array([con['h_min'], con['s_min'], con['v_min']]), 
                np.array([255,          con['s_max'], con['v_max']]))
            return cv.bitwise_or(mask0, mask1)
        else:
            return cv.inRange(img, 
                np.array([con['h_min'], con['s_min'], con['v_min']]), 
                np.array([con['h_max'], con['s_max'], con['v_max']]))
    

    def make_edges(self, img):
        convImg = cv.GaussianBlur(img, (5, 5), 0)
        return cv.Canny(convImg, 50, 50)


    def get_circle(self, color_idx, img):
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        self.centers[color_idx] = None
        self.contours[color_idx] = None
        if len(contours) > 0:
            biggest = max(contours, key=cv.contourArea)
            contourArea = cv.contourArea(biggest)
            ((x, y), radius) = cv.minEnclosingCircle(biggest)
            circleArea = np.pi * 2 * radius
            # at least 90% of the circle must be visible
            if circleArea * 0.9 < contourArea:
                m = cv.moments(biggest)
                self.centroids[color_idx] = (m["m10"] / m["m00"], m["m01"] / m["m00"])              
                self.centers[color_idx] = ((x, y), radius)


    def draw_circle(self, img, color_idx):
        center = self.centers[color_idx]
        if center:
            ((x, y), radius) = center
            (r, g, b) = cv.cvtColor(np.uint8([[self.colors[color_idx]]]), cv.COLOR_HSV2BGR)[0][0]
            cv.circle(img, (int(x), int(y)), int(radius + 10), ((int(r), int(g), int(b))))


    def process_img(self, img):
        """Process the image to find the interesting bits."""
        # Convert to HSV
        self.hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        if self.calibrating:
            hud_img = np.zeros_like(img)
            # filter for selected color
            self.masks[self.calibrating_color] = self.make_mask(self.hsv_img, self.calibrating_color)
            # make contour
            self.edges[self.calibrating_color] = self.make_edges(self.masks[self.calibrating_color])
            self.proc_img = cv.bitwise_and(img, img, mask=self.edges[self.calibrating_color])
            # get shapes
            self.get_circle(self.calibrating_color, self.edges[self.calibrating_color])
            # aside: render shape data
            self.draw_circle(hud_img, self.calibrating_color)
            self.proc_img += hud_img
        else:
            self.proc_img = np.zeros_like(img)
            hud_img = np.zeros_like(img)
            # Compute filtered image and analyze
            for i, con in enumerate(self.constraints):
                if con['is_on']:
                    # filter for selected color
                    self.masks[i] = self.make_mask(self.hsv_img, i)
                    # make contour
                    self.edges[i] = self.make_edges(self.masks[i])
                    # get shapes
                    self.get_circle(i, self.edges[i])
                    # aside: render shape data
                    self.draw_circle(hud_img, i)

            # Disjoin masks for global mask
            mask = np.zeros(img.shape[0:2], dtype=np.uint8)
            for i in range(0, len(self.constraints)):
                if self.constraints[i]['is_on']:
                    mask = cv.bitwise_or(mask, self.edges[i])

            self.proc_img = cv.bitwise_and(img, img, mask=mask)
            self.proc_img += hud_img


    constraintsFilename = 'constraints.json'

    def saveConstraints(self):
        with open(App.constraintsFilename, 'w') as f:
            json.dump(self.constraints, f, indent=4)


    def loadConstraints(self):
        if path.isfile(App.constraintsFilename):
            with open(App.constraintsFilename, 'r') as f:
                self.constraints = json.load(f)


    def run(self):
        """Main app run."""
        self.loadConstraints()
        self.computeColorsFromTrackbars()

        cap = cv.VideoCapture("float.mp4")
        if not cap.isOpened():
            print("No video capture devices were detected.")
            return

        img = None
        playing = True
        stepping = False
        while True:
            if playing or stepping:
                stepping = False
                success, inputImg = cap.read()
                if not success:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                else:
                    img = cv.resize(inputImg, (int(inputImg.shape[1] / inputImg.shape[0] * 320), 320))
                
            if self.calibrating:
                if self.refresh_calibration:
                    self.update_constraints_from_ui()
                    self.refresh_calibrationImage()
                    self.refresh_calibration = False

            self.process_img(img)
            if self.calibrating:
                cv.imshow("mask", self.masks[self.calibrating_color])

            display = np.hstack([img, self.proc_img])
            cv.imshow("video", display)

            key_pressed = cv.waitKey(13) & 0xFF
            # bail
            if key_pressed == ord('q') or key_pressed == 27:
                break
            # calibrate color filters 0-9
            elif (key_pressed >= ord('0') and
                  key_pressed <= ord('9')):
                new_color = key_pressed - ord('0')
                if self.calibrating_color != new_color or not self.calibrating:
                    self.calibrating_color = new_color
                    if not self.calibrating:
                        self.calibrating = True
                        self.make_calibrating_ui()
                    else:
                        self.set_ui_from_constraints()
                    self.refresh_calibration = True
            # scan mode
            elif key_pressed == ord('s'):
                if self.calibrating:
                    self.calibrating = False
                    self.destroy_calibrating_ui()
            # pause            
            elif key_pressed == ord(' '):
                playing = not playing
                stepping = False
            # video frame while paused
            elif key_pressed == ord('.'):
                playing = False
                stepping = True

        cap.release()
        cv.destroyAllWindows()

app = App()
app.run()
