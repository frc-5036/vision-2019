#!/usr/bin/env python3

import json
import time
import sys
from threading import Thread

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
import math
import datetime

class FPS:
	def __init__(self):
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		self._end = datetime.datetime.now()

	def update(self):
		self._numFrames += 1

	def elapsed(self):
		return (self._end - self._start).total_seconds()

	def fps(self):
		return self._numFrames / self.elapsed()

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None):
        self.outputStream = cameraServer.putVideo("stream", imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True
    def notifyError(self, error):
        self.outputStream.notifyError(error)

class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        self.webcam = camera
        self.webcam.setExposureManual(0)
        self.autoExpose = False
        self.prevValue = self.autoExpose
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        self.stream = cameraServer.getVideo()
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)
        self.name = name
        self.stopped = False

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if self.autoExpose:
                if(self.autoExpose != self.prevValue):
                    self.prevValue = self.autoExpose
                    self.webcam.setExposureAuto()
            else:
                if (self.autoExpose != self.prevValue):
                    self.prevValue = self.autoExpose
                    self.webcam.setExposureManual(0)
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        return self.timestamp, self.img

    def stop(self):
        self.stopped = True
    def getError(self):
        return self.stream.getError()

image_width = 256
image_height = 144

diagonalView = math.radians(58)

horizontalAspect = 16
verticalAspect = 9

diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))

green_blur = 7
#orange_blur = 27

# define range of green of retroreflective tape in HSV
lower_green = np.array([0,220,25])
upper_green = np.array([101, 255, 255])

def flipImage(frame):
    return cv2.flip( frame, -1 )

def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img,(blur_radius,blur_radius))
    return blur

def threshold_video(lower_color, upper_color, blur):

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask

def findTargets(frame, mask):
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    screenHeight, screenWidth, _ = frame.shape
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    image = frame.copy()
    if len(contours) != 0:
        image = findTape(contours, image, centerX, centerY)
    return image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    targets = []

    if len(contours) >= 2:
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        biggestCnts = []
        for cnt in cntsSorted:
            M = cv2.moments(cnt)
            hull = cv2.convexHull(cnt)
            cntArea = cv2.contourArea(cnt)
            hullArea = cv2.contourArea(hull)
            if (checkContours(cntArea, hullArea)):
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if(len(biggestCnts) < 13):
                    rotation = getEllipseRotation(image, cnt)

                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    boundingRect = cv2.boundingRect(cnt)
                    cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    cv2.circle(image, center, radius, (23, 184, 80), 1)

                    if [cx, cy, rotation, cnt] not in biggestCnts:
                         biggestCnts.append([cx, cy, rotation, cnt])


        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        for i in range(len(biggestCnts) - 1):
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        continue
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        continue
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                if [centerOfTarget, yawToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget])
    if (len(targets) > 0):
        networkTable.putBoolean("tapeDetected", True)
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

        currentAngleError = finalTarget[1]
        networkTable.putNumber("tapeYaw", currentAngleError)
    else:
        networkTable.putBoolean("tapeDetected", False)

    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image


def checkContours(cntSize, hullSize):
    return cntSize > (image_width / 6)

def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfTargetFromCamera = heightOfTarget - heightOfCamera

    '''
    d = distance
    h = height between camera and target
    a = angle = pitch

    tan a = h/d (opposite over adjacent)

    d = h / tan a

                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
    '''
    distance = math.fabs(heightOfTargetFromCamera / math.tan(math.radians(pitch)))

    return distance


def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)

def getEllipseRotation(image, cnt):
    try:
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        rotation = ellipse[2]
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        rotation = translateRotation(rotation, widthE, heightE)

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center = rect[0]
        rotation = rect[2]
        width = rect[1][0]
        height = rect[1][1]
        rotation = translateRotation(rotation, width, height)
        return rotation

#################### FRC VISION PI Image Specific #############
configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

"""Report parse error."""
def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

"""Read single camera configuration."""
def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

"""Read configuration file."""
def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

"""Start running the camera."""
def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    #Name of network table - this is how it communicates with robot. IMPORTANT
    networkTable = NetworkTables.getTable('devilVision')

    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)


    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    #Get the first camera

    webcam = cameras[0]
    cameraServer = streams[0]
    cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()

    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
    streamViewer = VideoShow(image_width,image_height, cameraServer, frame=img).start()
    #cap.autoExpose=True;
    tape = False
    fps = FPS().start()
    #TOTAL_FRAMES = 200;
    # loop forever
    while True:
        timestamp, img = cap.read()
        #frame = flipImage(img)
        frame = img
        if timestamp == 0:
            streamViewer.notifyError(cap.getError())
            continue
        if(networkTable.getBoolean("Driver", False)):
            cap.autoExpose = True
            processed = frame
        else:
            cap.autoExpose = False
            boxBlur = blurImg(frame, green_blur)
            threshold = threshold_video(lower_green, upper_green, boxBlur)
            processed = findTargets(frame, threshold)
        networkTable.putNumber("VideoTimestamp", timestamp)
        streamViewer.frame = processed
        fps.update()
        ntinst.flush()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))