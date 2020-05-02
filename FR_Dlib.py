# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:47:18 2020

@author: AVINASH SHUKLA
"""



def draw_border(img, pt1, pt2, color, thickness, r, d):
            x1,y1 = pt1
            x2,y2 = pt2
            # Top left
            cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)         
            # Top right
            cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            # Bottom left
            cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
            cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
            # Bottom right
            cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
            cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            
            
#%% IMporting Libraries

import os, re
import dlib
import cv2
import face_recognition
import numpy as np
import time
from imutils.video import FPS
from imutils import face_utils
from imutils.video import FileVideoStream

#%% storing data
# Declare all the list
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

#%%
known_face_encodings = []
known_face_names = []
known_faces_filenames = []
known_face_shapes = []
known_face_descriptions = []

# Walk in the folder to add every file name to known_faces_filenames
for (dirpath, dirnames, filenames) in os.walk('users/'):
    known_faces_filenames.extend(filenames)
    break

# Walk in the folder
for filename in known_faces_filenames:
    # Load each file
    face = face_recognition.load_image_file('users/' + filename)
    # Extract the name of each employee and add it to known_face_names
    known_face_names.append(re.sub("[0-9]",'', filename[:-4]))
    # Encode de face of every employee
    known_face_encodings.append(face_recognition.face_encodings(face)[0])
    # Converting the image to gray scale
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Get faces into webcam's image
    rects = detector(gray, 0)
    # Iterating over each image
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        face_chip = dlib.get_face_chip(face, shape)
        face_descriptor = facerec.compute_face_descriptor(face_chip)                
        face_descriptor = np.array(face_descriptor)
        known_face_descriptions.append(face_descriptor)
    
#%%  xbsaixb
video_capture = cv2.VideoCapture('http://192.168.43.1:2580/video')
fps = FPS().start()
process_this_frame = True
face_names = []
#%% the Core
while True:
    # Take every frame
    _, frame_1 = video_capture.read()
    frame = cv2.resize(frame_1, (0, 0),fx=0.5, fy=0.5 ,interpolation = cv2.INTER_AREA)
    #frame = dlib.load_rgb_image(frame_1)
    #size = frame_1.shape[1]/float( frame.shape[1] )
    # Process every frame only one time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        gray_test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     
        rects_test = detector(gray_test, 0)
        
        # getting names to list
        
        for (i, rect) in enumerate(rects_test):
            shape_test = predictor(gray_test, rect)
            face_chip_test = dlib.get_face_chip(frame, shape_test)
            face_descriptor_test = facerec.compute_face_descriptor(face_chip_test)                
            face_descriptor_test = np.array(face_descriptor_test)
            
            matches = face_recognition.compare_faces(known_face_descriptions, face_descriptor_test)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_descriptions, face_descriptor_test)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                
            face_names.append(name)
    process_this_frame = not process_this_frame
    
        
            
    #for (i, d) in enumerate(rects_test):
            #x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            #draw_border(frame_big, (x1, y1), (x2, y2), (248, 196, 113),4, 15, 10)
    
    
    for rect in rects_test:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        draw_border(frame, (x, y), (x + w, y + h), (91, 62, 239),2, 10, 5)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (99,112,236), 2)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        #for (sX, sY) in shape:
            #cv2.circle(frame_1, (sX, sY), 1, (248, 196, 113), -1)
            
        z = x - 15 if x - 15 > 15 else x + 15
        cv2.putText(frame, name, (x, z), cv2.FONT_HERSHEY_SIMPLEX,0.4, (122,212,149), 2)
        
    cv2.imshow('Video', frame)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

video_capture.release()
cv2.destroyAllWindows()

#http://192.168.43.1:2580/video
#fx=0.234375, fy=0.4166666666666667 for 1280x720
#fx=0.078125, fy=0.138888889  for 1920x1080