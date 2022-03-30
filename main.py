# GROUP-21 WAKE UP SLEEPY DRIVER USING ALERT
# Authors- Sanhita Paluskar & Ronisha Ronikkaraj

#importing the libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2


mixer.init()
#adding the sound alert to a alertSound variable
alertSound = mixer.Sound('alertSoundFile.wav')


# function for sending SMS starts
def sendSMS():
    # Download the helper library from https://www.twilio.com/docs/python/install
    from twilio.rest import Client

    # the following line needs your Twilio Account SID and Auth Token
    client = Client("AC1183dcc8503cb74bae2350f14def6ee1", "3bf0eccc8f423690e130b61cf57e2b62")

    # "from_" number: Twilio account number
    # "to" number: phone number with which we signed up for Twilio(authorised number)
    client.messages.create(to="+16132653830",
                           from_="+18646136148",
                           body="Hello XYZ! this is to inform you that your friend is sleepy while he is driving!")

# function for sendSMS ends

#function to calculate the threshold if eye is open ar not
def checkEye(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    er = (A + B) / (2.0 * C)
    return er

#setting threshhold value
eyethreshold= 0.25
#setting flag threshhold in frame
flag_threshold_frame = 25
#face detector library and predictor
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
#capture the video
capt = cv2.VideoCapture(0)
#initializinf the flag to 0
flag = 0
while True:
    # reads the captured video and returns false if no image has been captured from the video frame
    ret, frame=capt.read()
    # resize the frame window to 550 * 550 dimensions
    frame = imutils.resize(frame, height=550, width=550)
    # the captured is converted to grayscale
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    captures = face_detector(grayscale, 0)
    for capture in captures:
        shape = face_predictor(grayscale, capture)
        # converting the captured to NumPy Array
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[leftStart:leftEnd]
        rightEye = shape[rightStart:rightEnd]
        #calculating the eye ratio for both left and right eye
        leftER = checkEye(leftEye)
        rightER = checkEye(rightEye)
        eyeratio = (leftER + rightER) / 2.0
        # checking if the calculated eye ratio is less than initialized eye ratio
        if eyeratio < eyethreshold:
            flag += 1
            print(flag)
            #checking if the flag greater than the initialized flag- this is for time
            if flag >= flag_threshold_frame:
                #send SMS to friend
                #sendSMS()
                #play the alert sound
                alertSound.play()
                #add the text in the frame
                cv2.putText(frame, "DONT SLEEP!!", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
        else:
            flag = 0
    cv2.imshow("Eye-Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    #enter s or S to stop the code
    if key == ord("s") or key == ord("S"):
        break
cv2.destroyAllWindows()
capt.release()
