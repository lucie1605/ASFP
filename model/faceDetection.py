import cv2
import abc
import os
import time
from copy import deepcopy


class FaceDetection(abc.ABC):
    MODELS_PATH = os.path.dirname(__file__) + "\\dat_files\\"
    # BOXES_COLOUR = (0, 255, 0)  # green
    BOXES_COLOUR = (0,0,255) #red

    def __init__(self):
        self.coordinates_dico = {}
        self.faces_dico = {}
        self.outputImage = None
        self.formatedImage = None
        self.faces = []
        self.coordinates = []
        self.info = ""
        self.time_elapsed = 0

    def __format_image(self, convertToGray, appliedWidth):
        if appliedWidth:
            pass
            # if type(appliedWidth) != int:
            #     appliedWidth = 500
            # self.outputImage = imutils.resize(self.outputImage, width=appliedWidth)
        if convertToGray:
            self.formatedImage = cv2.cvtColor(self.outputImage, cv2.COLOR_BGR2RGB)

    def show_image_with_boxes(self):
        cv2.imshow(f"{len(self.faces)} faces detected", self.outputImage)
        cv2.waitKey(0)

    def get_image_with_boxes(self):
        return self.outputImage

    def get_info_about_computing(self):
        return f"{len(self.faces)} faces detected in {self.time_elapsed} seconds."

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
        self.faces_dico = {f"face_{i}":self.faces[i] for i in range(len(self.faces))}
        i=0
        for (x, y, w, h) in boxes:
            self.coordinates_dico[f"face_{i}"] = (x,y)
            i+=1
        for (x, y, w, h) in boxes:
            # draw the bounding box on our image
                cv2.rectangle(self.outputImage, (x, y), (x + w, y + h), FaceDetection.BOXES_COLOUR, 2)

        end = time.time()
        self.time_elapsed = end - start
        return self.faces

    def add_name_on_picture(self, face_id, name):
        # index_face = self.faces_dico[face_id]
        copy_image = deepcopy(self.outputImage)
        x, y = self.coordinates_dico[face_id]
        cv2.putText(copy_image, name, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, FaceDetection.BOXES_COLOUR, 2)
        return deepcopy(copy_image)
