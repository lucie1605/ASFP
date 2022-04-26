import os

from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
from model.faceRecognitionModel import FaceRecognitionModel
import ntpath


class ASFPModel:
    def __init__(self, detectionModelParam=2):
        self.peopleToIdentify = {}
        self.identified_persons = {}
        self.identified_persons_with_boxes = {}
        # init detection model
        self.detectionModel = FaceDetectionHOGLinearSVM()
        self.detectionModelParam = detectionModelParam
        # init recognition model
        self.recognitionModel = FaceRecognitionModel()

    def recognize_faces(self, peopleToIdentify, familyPhotos):
        # peopleToIdentify is a dico where key is the name, and value is the image of his face
        # familyPhotos is a directory path
        self.peopleToIdentify = peopleToIdentify
        for filename in os.listdir(familyPhotos):
            photo = familyPhotos + "/" + filename
            if os.path.isfile(photo):
                detectedFaces = self.detectionModel.get_detected_faces(photo, self.detectionModelParam)
                for face in detectedFaces:
                    person_identified = self.recognitionModel.get_identified_person(face, self.peopleToIdentify)
                    if person_identified and person_identified in self.identified_persons:
                        self.identified_persons[person_identified].append(photo)
                        self.identified_persons_with_boxes[person_identified]\
                            .append(self.detectionModel.get_image_with_boxes())
                    elif person_identified:
                        self.identified_persons[person_identified]=[photo]
                        self.identified_persons_with_boxes[person_identified]=[self.detectionModel.get_image_with_boxes()]

    def get_images_of_the_person_to_identify(self):
        return self.identified_persons

    def get_boxes_images_of_the_person_to_identify(self):
        return self.identified_persons_with_boxes