import os
import time
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

    def detect_members_of_family(self, member_photos_path):
        for filename in os.listdir(member_photos_path):
            photo = member_photos_path + "/" + filename
            if os.path.isfile(photo):
                detectedFaces = self.detectionModel.get_detected_faces(photo, self.detectionModelParam)
                detectedFaces_id = self.detectionModel.faces_dico
                name = ntpath.basename(photo).split(".")[0]
                self.peopleToIdentify[name] = list(detectedFaces_id.values())[0]
                print(f"[Detection] Member {name} detected in photo {filename} in {self.detectionModel.timeElapsed} seconds.")
        return self.peopleToIdentify

    def recognize_faces(self, family_photos_path):
        for filename in os.listdir(family_photos_path):
            photo = family_photos_path + "/" + filename
            if os.path.isfile(photo):
                detectedFaces = self.detectionModel.get_detected_faces(photo, self.detectionModelParam)
                detectedFaces_id = self.detectionModel.faces_dico
                print(f"[Detection] {len(detectedFaces_id)} faces detected in photo {filename} in {self.detectionModel.timeElapsed} seconds.")
                [self.__identify_face_in_picture(photo, face, face_id, filename) for face_id, face in detectedFaces_id.items()]

    def __identify_face_in_picture(self, photo, face, face_id, filename):
        start = time.time()
        person_identified = self.recognitionModel.get_identified_person(face, self.peopleToIdentify)
        image_with_name = self.detectionModel.add_name_on_picture(face_id, person_identified)
        end = time.time()
        name = person_identified
        print(
            f"[Recognition in {filename}] {face_id} is recognized as {person_identified} in {end - start} seconds.")
        if name and name in self.identified_persons:
            self.identified_persons[name].append(photo)
            self.identified_persons_with_boxes[name] \
                .append(image_with_name)
        elif name:
            self.identified_persons[name] = [photo]
            self.identified_persons_with_boxes[name] = [image_with_name]

    def get_images_of_the_person_to_identify(self):
        return self.identified_persons

    def get_boxes_images_of_the_person_to_identify(self):
        return self.identified_persons_with_boxes