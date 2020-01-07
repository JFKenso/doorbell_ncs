# USAGE
# python speed_estimation_dl.py --conf config/config.json

# import the necessary packages
from tempimage import TempImage
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from pyimagesearch.utils import Conf
from imutils.video import VideoStream
from imutils.io import TempFile
from imutils.video import FPS
from datetime import datetime
from threading import Thread
import numpy as np
import argparse
import dropbox
import imutils
import dlib
import time
import cv2
import os
import subprocess

def upload_file(tempFile, client, imageID):
	# upload the image to Dropbox and cleanup the tempory image
	print("[INFO] uploading {}...".format(imageID))
	path = "/{}.jpg".format(imageID)
	client.files_upload(open(tempFile.path, "rb").read(), path)
	tempFile.cleanup()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print("[SUCCESS] dropbox account linked")

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(conf["prototxt_path"], conf["model_path"])
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

print("Past sleep ... processing")

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
H = None
W = None

# keep the count of total number of frames
totalFrames = 0

# initialize the log file
logFile = None

avg = None

# loop over the frames of the stream
while True:
	# grab the next frame from the stream, store the current
	# timestamp, and store the new date
	frame = vs.read()
	ts = datetime.now()
	newDate = ts.strftime("%A %d %B %Y %I:%M:%S%p")

	output = frame.copy()
	motionFound = False

	# check if the frame is None, if so, break out of the loop
	if frame is None:
		break

	# if the log file has not been created or opened
	if logFile is None:
		# build the log file path and create/open the log file
		logPath = os.path.join(conf["output_path"], conf["csv_name"])
		logFile = open(logPath, mode="a")

		# set the file pointer to end of the file
		pos = logFile.seek(0, os.SEEK_END)

		# if we are using dropbox and this is a empty log file then
		# write the column headings
		if conf["use_dropbox"] and pos == 0:
			logFile.write("Year,Month,Day,Time,Speed (in MPH),ImageID\n")

		# otherwise, we are not using dropbox and this is a empty log
		# file then write the column headings
		elif pos == 0:
			logFile.write("Year,Month,Day,Time (in MPH),Speed\n")

	# Check if there's motion first, then run SSD
	if conf["motion_enabled"]:
		frameHeight, frameWidth = frame.shape[:2]
	
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
		# if the average frame is None, initialize it
		if avg is None:
			print ("[INFO] starting background model...")
			avg = gray.copy().astype("float")
			continue
	
		# accumulate the weighted average between the current frame and
		# previous frames, then compute the difference between the current
		# frame and running average
		cv2.accumulateWeighted(gray, avg, 0.5)
		frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

		# threshold the delta image, dilate the thresholded image to fill
		# in holes, then find contours on thresholded image
		thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
		#th3 = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		thresh = cv2.dilate(thresh, None, iterations=2)
		(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
		text = "na"

		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			#print("Contour: " + str(cv2.contourArea(c)))
			if cv2.contourArea(c) < conf["min_area"]:
				continue
	
			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
			motionHeight = x + h
			print ("Frame Width, Frame Height: " + str(frameWidth) + "," + str(frameHeight))
			print ("X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h))
			text =  "X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
			print ("Motion height: " + str(y + h))
			print ("Motion width: " + str(x + w))
			if y < conf["motionYThreshold"]:
				print ("Motion in the street ... ignoring")
				text =  "Motion in Street - X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
				continue
			else:
				print ("Motion within yard, alerting")
				text =  "Motion in Yard - X, Y, W, H: " + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
	
			motionFound = True

		# draw the text and timestamp on the frame
		cv2.putText(output, "Frontdoor Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(output, newDate, (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


	if motionFound:
		# resize the frame
		frame = imutils.resize(frame, width=conf["frame_width"])
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# initialize our list of bounding box rectangles returned by
		# either (1) our object detector or (2) the correlation trackers
		rects = []
	
		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % conf["track_object"] == 0:
			# initialize our new set of object trackers
			trackers = []
	
			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
			net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
			detections = net.forward()
			personFound = False
	
			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]
				idx = int(detections[0, 0, i, 1])

				label = CLASSES[idx]

				#print("Detected object: " + label + " confidence: " + str(confidence))
	
				# if the class label is not a car, ignore it
				if label != "person":
					continue

				print("Detected person: " + str(confidence))
				logFile.write("Detected person: " + str(confidence))
	
	
				# filter out weak detections by ensuring the `confidence`
				# is greater than the minimum confidence
				if confidence > conf["confidence"]:
					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")
					print("Person found: " + str(confidence) + " at " + str(box))
					logFile.write("Person found: " + str(confidence) + " at " + str(box))
	
					cv2.rectangle(frame, (startX,startY), (endX,endY), (0, 255, 0), 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					boxText = label + " " + str(int(confidence))
					cv2.putText(frame, boxText, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


					personFound = True
					
					
		if personFound:
			t = TempImage()
			cv2.imwrite(t.path, frame)

			if totalFrames % conf["motionSkipFrames"] == 0:
				try:
					with open(t.path, 'rb') as image:
						#subprocess.Popen(['echo "Ding Dong: " | mailx -s "' + label + ' Detected!" -A ' + t.path + ' YOUREMAILHERE@gmail.com'], shell=True)
						subprocess.Popen(['echo "Ding Dong: "  | mutt -a ' + t.path + ' -s "Ding Dong" -- YOUREMAILHERE@gmail.com'], shell=True)
	
				except:
					subprocess.Popen(['echo "exception caught: "  | mutt -s "Doorbell Exception Caught" -- YOUREMAILHERE@gmail.com'], shell=True)
					print ("Exception caught calling")
					logFile.write("Exception caught calling")
			
	
		# if the *display* flag is set, then display the current frame
		# to the screen and record if a user presses a key
		if conf["display"]:
			cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1

# check if the log file object exists, if it does, then close it
if logFile is not None:
	logFile.close()

# close any open windows
cv2.destroyAllWindows()

# clean up
print("[INFO] cleaning up...")
vs.stop()
