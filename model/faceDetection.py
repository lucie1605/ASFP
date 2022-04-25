import cv2
import imutils
import abc
import os
import time


class FaceDetection(abc.ABC):
    MODELS_PATH = os.path.dirname(__file__) + "\\dat_files\\"

    def __init__(self):
        self.outputImage = None
        self.formatedImage = None
        self.faces = []
        self.info = ""
        self.timeElapsed = 0

    def __format_image(self, convertToGray, appliedWidth):
        if appliedWidth:
            if type(appliedWidth) != int:
                appliedWidth = 500
            self.outputImage = imutils.resize(self.outputImage, width=appliedWidth)
        if convertToGray:
            self.formatedImage = cv2.cvtColor(self.outputImage, cv2.COLOR_BGR2RGB)

    def show_image_with_boxes(self):
        cv2.imshow(f"{len(self.faces)} faces detected", self.outputImage)
        cv2.waitKey(0)

    def get_image_with_boxes(self):
        return self.outputImage

    def get_info_about_computing(self):
        return f"{len(self.faces)} faces detected in {self.timeElapsed} seconds."

    @abc.abstractmethod
    def get_faces_rectangles(self, modelParam=2):
        pass

    def get_detected_faces(self, imagePath,  modelParam=2, convertToGray=False, appliedWidth=None):
        start = time.time()
        self.__format_image(convertToGray, appliedWidth)
        self.formatedImage = cv2.imread(imagePath)
        self.outputImage = self.formatedImage

        boxes = self.get_faces_rectangles(modelParam)
        # slice image to have only the detected face
        self.faces = [self.outputImage[y:y + h, x:x + w] for (x, y, w, h) in boxes]
        for (x, y, w, h) in boxes:
            # draw the bounding box on our image
            cv2.rectangle(self.outputImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

        end = time.time()
        self.timeElapsed = end - start
        return self.faces
