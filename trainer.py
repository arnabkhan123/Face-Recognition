import cv2
import os
import pickle
import time
import numpy as np
from PIL import Image
from collections import OrderedDict


# set a scade Path for face detection using facial features
cascadePath = 'D:\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

# set algorithm for identifying a face
#recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.FisherFaceRecognizer_create()
#recognizer = cv2.createEigenFaceRecognizer()
imageSize = (200, 200)
# Captures a single image from the camera and returns it in PIL format


def get_image(camera):
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im


# a utility function that will --
def get_images_and_labels(path):
    images = []
    labels = []
    for root, dirs, files in os.walk(path):
        for image_name in files:
            image_pil = Image.open(os.path.join(root, image_name)).convert('L')
            image = np.array(image_pil)
            # detect face from each image
            images.append(image)
            label = os.path.dirname(os.path.join(root, image_name)).replace(path, '')
            labels.append(int(label))
            print("image name:" + image_name + " label :" + label)
    return images, labels


def addNewDataset(path):
	name = input("Enter your name..\n")
	i = 1
	camera_port = 0
	camera = cv2.VideoCapture(camera_port)
	mapping = pickle.load(open("mapping.p","rb"))
	if(len(mapping) == 0):
		lastInsertedKey = -1
	else:
		lastInsertedKey = list(mapping.keys())[-1]
	print("Last inserted key was" + str(lastInsertedKey))
	mapping.update({(lastInsertedKey+1): name})
	pickle.dump(mapping,open("mapping.p","wb"))
	if not os.path.exists(path + str((lastInsertedKey+1))):
		os.mkdir(path + str((lastInsertedKey+1)))
	while (i <= 50):
		while True:
			image = get_image(camera)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			detectedFace = faceCascade.detectMultiScale(
				image,
				scaleFactor=1.1,
				minNeighbors=5,
				minSize=(180,180),
				flags=cv2.CASCADE_SCALE_IMAGE
				)
			if len(detectedFace) == 0:
				continue
			for (x, y, w, h) in detectedFace:
				image = image[y:y + h, x: x + h]
				image = cv2.resize(image,imageSize)
				cv2.imwrite(path + str((lastInsertedKey+1))+"/" + str(i) + ".jpg", image)
				cv2.imshow("image", image)
				cv2.waitKey(100)
				time.sleep(0.100)
				cv2.destroyAllWindows()
				i = i + 1
				break
			break
	del(camera)


def trainTheModel():
	path = os.getcwd() + '/image_database/'
	addNewDataset(path)
	images, labels = get_images_and_labels(path)
	# train the algorithm with the images
	recognizer.train(np.array(images), np.array(labels))
	recognizer.write("mytrainingdata.xml")

# main function:


if __name__ == '__main__':
	trainTheModel()
