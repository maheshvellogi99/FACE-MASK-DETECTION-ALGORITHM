# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import keras
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Email Configuration
EMAIL_SENDER = "maheshvellogi99@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "xpse tcnl knqq tefe"    # Replace with your app password
EMAIL_RECEIVER = "youngonion.m@gmail.com"  # Replace with receiver email

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

# Variable to track last announcement time
last_announcement_time = 0
ANNOUNCEMENT_COOLDOWN = 5  # Seconds between announcements

# Dictionary to track mask violations for each face
face_violations = {}

def send_email_alert(image_path):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = 'Mask Violation Alert'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        # Add text to email
        text = f"Mask violation detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg.attach(MIMEText(text, 'plain'))

        # Add image to email
        with open(image_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename='violation.jpg')
            msg.attach(img)

        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            
        print(f"[INFO] Email alert sent successfully")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {str(e)}")

def save_violation_image(frame, face_box):
    # Create violations directory if it doesn't exist
    if not os.path.exists('violations'):
        os.makedirs('violations')
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'violations/violation_{timestamp}.jpg'
    
    # Save the image
    cv2.imwrite(filename, frame)
    return filename

def track_face_violation(face_id, mask_detected):
    if face_id not in face_violations:
        face_violations[face_id] = {'violations': 0, 'last_violation_time': time.time()}
    
    if not mask_detected:
        face_violations[face_id]['violations'] += 1
        face_violations[face_id]['last_violation_time'] = time.time()
        return face_violations[face_id]['violations']
    else:
        face_violations[face_id]['violations'] = 0
        return 0

def get_face_id(box):
    # Simple face identification based on position
    # This is a basic implementation - you might want to use more sophisticated face recognition
    return f"{box[0]}_{box[1]}"

def announce_status(mask_detected):
    global last_announcement_time
    current_time = time.time()
    
    if current_time - last_announcement_time >= ANNOUNCEMENT_COOLDOWN:
        if mask_detected:
            engine.say("Entry Permission granted, welcome")
        else:
            engine.say("Please wear a mask. Entry denied without mask")
        engine.runAndWait()
        last_announcement_time = current_time

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = keras.models.load_model("mask_detector.keras")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 800 pixels for better quality
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Track violations for this face
		face_id = get_face_id((startX, startY, endX, endY))
		violations = track_face_violation(face_id, mask > withoutMask)
		
		# Check if we need to send an alert
		if violations >= 5:
			image_path = save_violation_image(frame, (startX, startY, endX, endY))
			send_email_alert(image_path)
			# Reset violations after sending alert
			face_violations[face_id]['violations'] = 0

		# Announce status based on mask detection
		announce_status(mask > withoutMask)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		# Get text size to create background rectangle
		(text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
		
		# Draw background rectangle for text with padding
		padding = 5
		cv2.rectangle(frame, 
			(startX, startY - text_height - padding), 
			(startX + text_width + padding*2, startY + padding), 
			color, -1)
		
		# Draw text with white color and shadow for better contrast
		# Shadow effect
		cv2.putText(frame, label, (startX + 1, startY - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
		# Main text
		cv2.putText(frame, label, (startX, startY - 5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
		
		# Draw bounding box with thicker lines
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

	# show the output frame with enhanced quit text
	# Background for quit text
	quit_text = "Press 'q' or 'ESC' to quit"
	(quit_width, quit_height), _ = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
	cv2.rectangle(frame, (5, 5), (quit_width + 15, quit_height + 10), (0, 0, 0), -1)
	
	# Quit text with shadow
	cv2.putText(frame, quit_text, (8, 18),
		cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
	cv2.putText(frame, quit_text, (7, 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key or ESC key was pressed, break from the loop
	if key == ord("q") or key == 27:  # 27 is the ASCII code for ESC
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()