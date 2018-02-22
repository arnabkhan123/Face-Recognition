
#import modules
import cv2, os, pickle
import numpy as np
from PIL import Image

#set a scade Path for face detection using facial features
cascadePath = 'D:\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

#set algorithm for identifying a face
#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.FisherFaceRecognizer_create()
#recognizer = cv2.face.createEigenFaceRecognizer()

mapping = pickle.load(open("mapping.p","rb"))
 
# Captures a single image from the camera and returns it in PIL format
def get_image(camera):
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im

def startRecognizing():
    #save the images for future reference.
    recognizer.read("mytrainingdata.xml")
    match = False
    time = 0
    # Set the port of web cam
    camera_port = 0
     
    #Number of frames to throw away while the camera adjusts to light levels
    ramp_frames = 30
     
    # Now we can initialize the camera capture object with the cv2.VideoCapture.
    camera = cv2.VideoCapture(camera_port)

    while True:
        
        cameraCapture = get_image(camera)
        grayImage = cv2.cvtColor(cameraCapture, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                grayImage,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(180, 180),
                flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            predictImage = grayImage[y: y + h, x: x + w]
            predictImage = cv2.resize(predictImage, (200,200))
            label, conf = recognizer.predict(predictImage)
            if(conf<10000):
                cv2.rectangle(cameraCapture, (x, y), (x + w, y + h),(0,255,0),1)
                cv2.putText(cameraCapture,'%s : %.0f' % (mapping.get(label), conf),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0))
            else:
                cv2.rectangle(cameraCapture, (x, y), (x + w, y + h),(0,0,255),1)
                cv2.putText(cameraCapture,'Not Identified',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255))
        cv2.imshow("Recognizing..",cameraCapture)
        cv2.waitKey(60)    

    del(camera)

#main function:
if __name__ == '__main__':
    startRecognizing()
