import os
import time
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
from model.faceRecognitionModel import FaceRecognitionModel


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
                detectedFaces_id = self.detectionModel.faces_dico
                print(f"[Detection] {len(detectedFaces)} faces detected in photo {filename} in {self.detectionModel.timeElapsed} seconds.")
                # one person can be present only one time on a photo
                people_identified_on_photo = {}
                for face_id, face in detectedFaces_id.items():
                    start = time.time()
                    person_identified = self.recognitionModel.get_identified_person(face, self.peopleToIdentify)
                    end = time.time()
                    print(f"[Recognition] {face_id} is recognized as {person_identified} in {end-start} seconds.")
                    people_identified_on_photo[person_identified] = face_id
                #for name, face_id in people_identified_on_photo.items():
                    image_with_name = self.detectionModel.add_name_on_picture(face_id, person_identified)
                    name = person_identified
                # for name in people_identified_on_photo.keys():
                    if name and name in self.identified_persons:
                        self.identified_persons[name].append(photo)
                        self.identified_persons_with_boxes[name]\
                            .append(image_with_name)
                    elif name:
                        self.identified_persons[name]=[photo]
                        self.identified_persons_with_boxes[name]=[image_with_name]

    def get_images_of_the_person_to_identify(self):
        return self.identified_persons

    def get_boxes_images_of_the_person_to_identify(self):
        return self.identified_persons_with_boxes