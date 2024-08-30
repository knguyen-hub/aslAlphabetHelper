import cv2
import numpy as np

import onnx
import onnxruntime as ort

import random

from time import time

def nothing1(x):
    pass

def get_random_response():
    rand_value = random.random()
    if rand_value < 0.25:
        return 'Good job!'
    elif rand_value < 0.5:
        return 'Correct!'
    elif rand_value < 0.75:
        return 'Keep it up!'
    elif rand_value < 1.0:
        return 'Right!'

is_random = False
def randomize_button_callback(state):
    if state == 1:
        is_random = True
    else:
        is_random = False

cv2.namedWindow('Sign Language Translator')
cv2.createTrackbar('X', 'Sign Language Translator', 50, 1180, nothing1)
cv2.createTrackbar('Y', 'Sign Language Translator', 150, 620, nothing1)
cv2.createTrackbar('Size', 'Sign Language Translator', 130, 300, nothing1)
cv2.createTrackbar('Quiz', 'Sign Language Translator', 1, 1, nothing1)
cv2.createTrackbar('Random', 'Sign Language Translator', 0, 1, nothing1)


ort_session = ort.InferenceSession("signlanguage1.onnx")

index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')

cap = cv2.VideoCapture(0)

help_frame = np.zeros((620, 1180, 3), np.uint8)
cv2.rectangle(help_frame, (0,0), (1180, 620), (240,240,240), -1)

cv2.putText(help_frame, "Welcome to the ASL Alphabet Application!",
            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "This program uses a machine learning model to classify sign language letters.",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "Place your hand within the black square shown on the screen",
            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "Adjust this square by using the X, Y, and Size trackbars",
            (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "For best results, use with a moderately clear background and lighting",
            (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "The program displays which letter to sign in the upper right corner.",
            (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "To randomize the order, toggle the Random slider.",
            (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

cv2.putText(help_frame, "Press the spacebar to close this window",
            (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)

run_program = True
# check if camera is available
while (not cap.isOpened()):
    message_img = np.zeros((620,1180,3), np.uint8)

    cv2.putText(message_img, "Camera not available.", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), thickness=2)
    cv2.putText(message_img, "Make sure your camera is not already in use, then press the spacebar.",
                (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    cv2.putText(message_img, "The program will then begin shortly.", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    
    cv2.imshow("Camera not available", message_img)
    k = cv2.waitKey(5) & 0xFF
    if k == 32:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cv2.destroyWindow("Camera not available")
            break
    elif k == 27:
        run_program = False
        break

quiz_index = 0
is_correct = False
initial_time = time()
quiz_initial_time = time()

showing_help = True
index_list = []

while(run_program):

    xCrop = cv2.getTrackbarPos('X', 'Sign Language Translator')
    yCrop = cv2.getTrackbarPos('Y', 'Sign Language Translator')
    zoom = cv2.getTrackbarPos('Size', 'Sign Language Translator')
    quiz = cv2.getTrackbarPos('Quiz', 'Sign Language Translator')
    is_random = cv2.getTrackbarPos('Random', 'Sign Language Translator')
    
    # get and crop frame
    _, frame = cap.read()

    
    display = frame
    x = frame[yCrop: yCrop+zoom, xCrop: xCrop+zoom]
    
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (28, 28))

    # find mean and standard deviation of pixel values
    mean, std = x.mean(), x.std()

    # normalize pixel values
    x = (x - mean) / std
    np.clip(x, -1, 1, x)
    x = (x + 1.0) / 2.0

    x = x.reshape(1, 1, 28, 28).astype(np.float32)

    # put through network
    y = ort_session.run(None, {'input': x})[0]

    # get letter
    index = np.argmax(y, axis=1)
    letter = index_to_letter[int(index)]

    # quiz program
    if quiz == 1:   
        quiz_letter = index_to_letter[quiz_index]
        if len(index_list) == 24:
            index_list.clear()
        index_list.append(quiz_index)
        # check if letter is correct
        if quiz_letter == letter and not is_correct:
            quiz_initial_time = time()
            is_correct = True
            response = get_random_response()

        if time() < quiz_initial_time + 1 and is_correct:
            display = cv2.flip(display, 1)
            cv2.putText(display, response, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), thickness=2)
            display = cv2.flip(display, 1)

        
        elif time() > quiz_initial_time + 1 and is_correct:
            if is_random == 1:

                
                unique_letter = False
                while not unique_letter:
                    if quiz_index in index_list:
                        quiz_index  = int(random.random() * 24)
                    else:
                        unique_letter = True

                
            else :
                quiz_index += 1

            if quiz_index > 23:
                quiz_index = 0
            is_correct = False

        
        
        display = cv2.flip(display, 1)
        cv2.putText(display, quiz_letter, (500, 125), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), thickness=2)

    # detecting program
    else:
        display = cv2.flip(display, 1)
        cv2.putText(display, letter, (100, 125), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 0, 0), thickness=2)

    # display
    display = cv2.flip(display, 1)
    cv2.rectangle(display, (xCrop, yCrop), (xCrop + zoom, yCrop + zoom), (0,0,0), 3)
    display = cv2.flip(display, 1)

    if time() < initial_time + 3:
        cv2.putText(display, "Press h for help, esc to quit", (10, 20), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2)
        
    
    cv2.imshow("Sign Language Translator", display)

    
    k = cv2.waitKey(5) & 0xFF

    if k == 104 or k == 72:
        cv2.imshow("Help", help_frame)
        showing_help = True
    if k == 32 and showing_help:
        cv2.destroyWindow("Help")
        
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()
