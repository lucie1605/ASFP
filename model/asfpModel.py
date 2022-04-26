from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
from model.faceRecognitionModel import FaceRecognitionModel
import ntpath


class ASFPModel:
    def __init__(self, detectionModelParam):
        self.persons_to_identify = {}
        self.identified_persons = {}
        self.identified_persons_with_boxes = {}
        # init detection model
        self.detectionModel = FaceDetectionHOGLinearSVM()
        self.detectionModelParam = detectionModelParam
        # init recognition model
        self.recognitionModel = FaceRecognitionModel()

    def recognize_faces(self, personsToIdentify, familyPhotos):
        # personsToIdentify is a list of image Path
        # the name of the file is the name of the person to identify
        # persons_to_identify is a dico where key is the name, and value the path of the image
        self.persons_to_identify = {ntpath.basename(imagePath):imagePath for imagePath in personsToIdentify }
        faces = []
        for photo in familyPhotos:
            faces.append(self.detectionModel.get_detected_faces(photo, self.detectionModelParam))
            for face in faces:
                person_identified = self.recognitionModel.get_identified_person(face, self.persons_to_identify)
                if person_identified:
                    self.identified_persons[person_identified].append(photo)
                    self.identified_persons_with_boxes[person_identified].append(self.get_image_with_boxes())

    def get_images_of_the_person_to_identify(self, person_to_identify):
        return self.identified_persons[person_to_identify]

    def get_boxes_images_of_the_person_to_identify(self, person_to_identify):
        return self.identified_persons_with_boxes[person_to_identify]