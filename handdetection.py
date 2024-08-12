import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os
import uuid
import numpy as np
import pandas as pd
import csv
import tensorflow as tf

from classify_single_img import classify_single_img

saved_model = "asl_class_50.001.keras"


def findImageLandmarks(image_path, save_annotated_img=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        static_image_mode = True
        #flip for correct hand
        image = cv2.flip(cv2.imread(image_path), 1)

        # BGR to RGB
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        #Todo: Change this into returning the landmarklist itself so we can put all the image's landmark lists together into one dataframe/csv
        # print(results.multi_hand_landmarks)



        if save_annotated_img:
            print('Handedness:', results.multi_handedness)

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.imwrite(
                    image_path.replace(".jpg", "") + "_annotated.png", cv2.flip(annotated_image, 1))
                
        return results.multi_hand_landmarks


def createLandmarkDS(ds, save_CSV=False):
    labels = os.listdir(ds)
    labels.remove('.DS_Store')
    landmarks = {}

    for label in labels:
        print(label)
        images = os.listdir(str(ds + "/" + label))
        for image in images:
            image_path = str(ds + "/" + label + "/" + image)
            lm = findImageLandmarks(image_path, False)
            if lm != None:
                landmarks[image] = lm
        break      
    
    print(landmarks)

            

        



# createLandmarkDS("/Users/jasonkuchtey/Desktop/asl_data/archive/asl_alphabet_train/asl_alphabet_train")



def liveDetect(output_dir=None, save_feed=False, predict=False, class_names=None, image_size=None, draw=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if save_feed:
        os.mkdir(output_dir)


    # initialize video capture
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            ret, frame = cap.read()

            #Convert from BGR to RGB in order to use with mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


            # Rendering results
            # If there is a hand in the feed
            if results.multi_hand_landmarks:
                # print(results.multi_hand_landmarks)
                
                if predict:
                    model = tf.keras.models.load_model(saved_model)
                    resized = cv2.resize(image, dsize=(image_size, image_size), interpolation=cv2.INTER_LINEAR)
                    classify_single_img(resized, image_size, model, class_names, npimage=True)
                    

                if draw:
                    # For every hand
                    for num, hand in enumerate(results.multi_hand_landmarks):
                        # Draw every landmark for that hand
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                                                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                                )

                fcount = 0
                if save_feed:
                    cv2.imwrite(os.path.join(output_dir, '{}.jpg'.format(uuid.uuid1())), image)
                    fcount += 1

            #Displays webcam feed
            
            cv2.imshow('Hand Tracking', image)

            # Kills webcam feed when q key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



liveDetect(draw=True)
# findImageLandmarks("f1.jpg", save_annotated_img=False)