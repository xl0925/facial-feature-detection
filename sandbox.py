from camera import Camera


path = "C:\\Users\\xil14\\PycharmProjects\\webcam"
cascade_path = path + '\\resources\\haarcascade_frontalface_default.xml'
tolerance = 10
category = 'Attractive Female'
learner_path = path + '\\resources\\models\\attractive_female_resnet50.pkl'


def feed(self):
    def extractFaceCoords(img, cascade, tolerance):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_coords = cascade.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))
        return face_coords


