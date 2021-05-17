from utils import reference_anchors, average_maximum_sup

from threading import Thread
import cv2
import tensorflow as tf

import imutils
import time

# The WebcamVideoStream function is from
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/

class WebcamVideoStream:
	def __init__(self, src=0):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True

vs = WebcamVideoStream(src=0).start()

start_time = time.time()
x = 1 # displays the frame rate every 1 second
counter = 0
fps = 0

face_model = tf.keras.models.load_model('weights/saved_model/face')
print("face_model loaded")

while True:
	frame = vs.read() # 720x1280x3 heightxwidthxchannels
	H, W, C = frame.shape

	color = (0, 0, 255)
	# Draw a rectangle where the image will get cropped
	img = cv2.rectangle(frame, (180, 1), (1100, 719), color, 2)
	img = frame[:, :, [2, 1, 0]] # change to rgb
	# frame = img.numpy()

	# Resize frame to render a smaller image and have better performance
	frame = imutils.resize(img, width=400) # 225x400x3

	img = tf.image.crop_to_bounding_box(img, 0, 180, 720, 920) # 720x920x3

	padding = [[100, 100], [0, 0], [0, 0]]

	img = tf.cast(img / 255, tf.float32)

	img = tf.pad(img, padding) # 920x920x3

	img = tf.image.resize(img, [128, 128])
	imgs = tf.expand_dims(img, axis=0)
	
	# print("prediction started")
	anchor_predictions, class_predictions = face_model(imgs, training=False)
	# print("prediction done")
	output_detections, final_detections = average_maximum_sup(class_predictions, anchor_predictions, reference_anchors)

	current_detection = output_detections[0]
	score = 0

	cut_images = []
	cut_images_coords = []

	if current_detection != []:
		for box in current_detection:
			score = box[0]
			x1 = int(box[1] * 920) + 180
			x1 = (x1 / 1280) * 400
			x1 = int(x1)

			y1 = int(box[2] * 920) - 100
			y1 = (y1 / 720) * 225
			y1 = int(y1)

			x2 = int(box[3] * 920) + 180
			x2 = (x2 / 1280) * 400
			x2 = int(x2)

			y2 = int(box[4] * 920) - 100
			y2 = (y2 / 720) * 225
			y2 = int(y2)

			color = (0, 0, 255)
			frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

	font = cv2.FONT_HERSHEY_DUPLEX
	text = f"FPS: {fps}"
	cv2.putText(frame, text, (10, 10), font, 0.5, (255, 255, 255), 1)

	frame = frame[:, :, [2, 1, 0]]
	cv2.imshow("Frame", frame)

	counter += 1
	if (time.time() - start_time) > x :
		fps = counter / (time.time() - start_time)
		counter = 0
		start_time = time.time()

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
vs.stop()