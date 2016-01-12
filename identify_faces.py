

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import os
import multiprocessing

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Person(object):
    last_frame_seen = 1000000
    
    def __init__ self, name, class_id):
        self.name = name
        self.class_id = class_id
        self.last_frame_seen

    def welcome_message(self):
        time = time.strftime('%H:%M')

def detect_faces(image, frames_passed_in_session, output):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=4, minSize=(60,60), maxSize=(300, 300))
    faces_found = []
    for (x, y, w, h) in faces:
        print("face found!")
        face_roi = image[y:y+h, x:x+w]
        faces_found.append(face_roi)
        
    output.put(faces_found)

def identify_faces(faces_found):
    identifed_faces = []
    for each_face in faces_found:
        pass

    return identifed_faces

def welcome_message(person_identifed):
    #os.system("espeak 'Movement Detected!'")
    intro_speech = 'Hello ' + person_identifed + ', how are you?'
    weather = 'Here is the forecast for the rest of the day'

def main():

    people_stored = [Person('Jack', 0), Person('Rina', 1)]

    camera = PiCamera()
    resolution = ((1280, 720)) #(640, 480)  (1920, 1080)
    fps = 10
    camera.resolution = resolution
    camera.framerate = fps
    rawCapture = PiRGBArray(camera, size=resolution)
    
    time.sleep(0.1)

    background_image = None
    frames_passed_in_session = 0
    num_frames_set_bg = 25
    images_from_segment = []
    
    output = multiprocessing.Queue()
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        image = frame.array
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        float_smooth_gray_image = np.float32(smooth_gray_image)

        if frames_passed_in_session == 0:
            background_image = float_smooth_gray_image
        else:
            cv2.accumulateWeighted(float_smooth_gray_image, background_image, 0.1)

        background_difference = cv2.absdiff(float_smooth_gray_image, background_image)
        mask_difference = cv2.bitwise_and(background_difference, background_image)
        abs_mask_difference = np.uint8(mask_difference)

        frames_passed_in_session += 1
 
        if frames_passed_in_session > num_frames_set_bg: # time delay for bg to average

            ret, thresh = cv2.threshold(abs_mask_difference, 50, 255, cv2.THRESH_BINARY)
            num_pixels_different = np.sum(thresh)
            print (num_pixels_different)

            if num_pixels_different > 10000:
                images_from_segment.append((frames_passed_in_session, image))
                
        else:
            print (frames_passed_in_session)

        if frames_passed_in_session > num_frames_set_bg and frames_passed_in_session % fps == 0:

            processes = [multiprocessing.Process(target=detect_faces, args=(each_image, i, output)) for i, each_image in images_from_segment]

            for p in processes:
                p.start()
            for p in processes:
                p.join()
            result = [output.get() for p in processes]
            images_from_segment = []
            
            faces_found = []
            for face_list in result:
                faces_found.extend(face_list)
            
            people_identified_ids = list(set(identify_faces(faces_found)))

            for person_id in people_identified_ids:
                for person in people_stored:
                    if person_id == person.class_id and frames_passed_in_session - person.last_frame_seen > 36000:
                        person.welcome_message()
                        person.last_frame_seen = frames_passed_in_session
            

    	#cv2.imshow("Frame", abs_mask_difference)
    	#key = cv2.waitKey(1) & 0xFF

    	rawCapture.truncate(0)

    	#if key == ord("q"):
    		#break

if __name__ == "__main__":
    main()
