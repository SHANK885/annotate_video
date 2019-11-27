'''
python annotate_from_videos.py -v ../data/drone.mp4 -l Car

'''
from imutils.video import  VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import csv
import  os
import copy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v",
                "--video",
                type=str,
                help="path to input video files")

ap.add_argument("-t",
                "--tracker",
                type=str,
                default="csrt",
                help="OpenCV object tracker type")
ap.add_argument("-l",
                "--label",
                type=str,
                default="None",
                help="label of annotating object")

args = vars(ap.parse_args())

# extract the opencv version information
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using opencv 3.2 or less we can use a special factory
# function to create our object Tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 or above, we need to explicitly call the
# appropriate object tracker constructor

else:
    # initialize, a dictionary that maps string to their corresponding
    # OpenCV object tracker constructor
    OPENCV_OBJECT_TRACKERS = {"csrt": cv2.TrackerCSRT_create,
                              "kcf": cv2.TrackerKCF_create,
                              "boosting": cv2.TrackerBoosting_create,
                              # "mil": cv2.TrackerMIL_create,
                              # "tld": cv2.TrackerTLD_create,
                              # "medianflow": cv2.TrackerMedianFlow_create,
                              # "mosse":cv2.TrackerMOSSE_create
                             }
    # grab the appropriate object tracker using our dictionary of
  	# OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates we are going to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream from camera(0)...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise grab a referance to the provided video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
im_count = len(os.listdir("../images/"))
f_num = 0
images = []

if im_count == 0:
    # initialize annotation files
    header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    with open('../annotation/labels.csv', 'w', newline='') as csvWriter:
        wr = csv.writer(csvWriter, quoting=csv.QUOTE_ALL)
        wr.writerow(header)

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream of VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    raw_frame = copy.copy(frame)

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so that we can process it faser) and grab the
    # frame dimensions
    #frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    # print("frame shape:", frame.shape[:2])

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding boc coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if out tracking was success
        if success:
            # save the frame
            (x, y, w, h) = [int(v) for v in box]

            # print("f_num: ", f_num)
            if f_num%10 == 0:
                im_count += 1
                im_name = "img_{}.jpg".format(im_count)
                path = os.path.join("../images/", im_name)
                print("bounding box: {}".format((x, y, w, h)))
                print(path)
                cv2.imwrite(path, raw_frame)
                images.append([im_name, w, h, args["label"], x, y, x+w, y+h])

            if f_num > 1000:
                f_num = 0
            f_num += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # update the fps counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on the frames
        info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps()))
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame,
                        text,
                        (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s") or key == ord("S"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI
        initBB = cv2.selectROI("Frame",
                               frame,
                               fromCenter=False,
                               showCrosshair=True)

        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the 'q' key was pressed, break from the loop
    elif key == ord("q") or key == ord("Q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()

with open('../annotation/labels.csv', 'a', newline='') as csvWriter:
    wr = csv.writer(csvWriter, quoting=csv.QUOTE_ALL)
    for im in images:
        wr.writerow(im)

print()
print("Quitting.........")
print("{} images saved and annotated".format(im_count))
