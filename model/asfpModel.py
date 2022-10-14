import os
import time
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
from model.faceRecognitionModel import FaceRecognitionModel
import ntpath


class ASFPModel:
    def __init__(self, detectionModelParam=2):
        self.peopleToIdentify = {}
        self.identified_people = {}
        self.identified_people_with_boxes = {}
        self.photo_without_anybody = []
        # init detection model
        self.detectionModel = FaceDetectionHOGLinearSVM()
        self.detectionModelParam = detectionModelParam
        # init recognition model
        self.recognitionModel = FaceRecognitionModel()

    def detect_members_of_family(self, member_photos_path):
        for filename in os.listdir(member_photos_path):
            photo = member_photos_path + "/" + filename
            if os.path.isfile(photo):
                self.detectionModel.get_detected_faces(photo, self.detectionModelParam)
                detected_faces_id = self.detectionModel.faces_dico
                name = ntpath.basename(photo).split(".")[0]
                if not detected_faces_id:
                    print(
                        f"[Family Member Detection] FAIL to detect {name} in photo {filename}"
                        f" in {self.detectionModel.time_elapsed} seconds.")
                else:
                    self.peopleToIdentify[name] = list(detected_faces_id.values())[0]
                    print(f"[Family Member Detection] {name} detected in photo {filename}"
                          f" in {self.detectionModel.time_elapsed} seconds.")
        print("*"*100)
        return self.peopleToIdentify

    def recognize_faces(self, family_photos_path):
        for filename in os.listdir(family_photos_path):
            photo = family_photos_path + "/" + filename
            if os.path.isfile(photo):
                self.detectionModel.get_detected_faces(photo, self.detectionModelParam)
                detected_faces_id = self.detectionModel.faces_dico
                print(f"[Detection in {filename}] {len(detected_faces_id)} faces detected."
                      f" in {self.detectionModel.time_elapsed} seconds.")
                nb_detected_people = 0
                for face_id, face in detected_faces_id.items():
                    person_identified = self.__identify_face_in_picture(photo, face, face_id, filename)
                    if person_identified:
                        nb_detected_people+=1
                if not nb_detected_people:
                    self.photo_without_anybody.append(self.detectionModel.get_image_with_boxes())
            print("*" * 100)

    def __identify_face_in_picture(self, photo, face, face_id, filename):
        start = time.time()
        person_identified = self.recognitionModel.get_identified_person(face, self.peopleToIdentify)
        image_with_name = self.detectionModel.add_name_on_picture(face_id, person_identified)
        end = time.time()
        print(
            f"[Recognition in {filename}] {face_id} is recognized as {person_identified} in {end - start} seconds.")
        if person_identified and person_identified in self.identified_people:
            self.identified_people[person_identified].append(photo)
            self.identified_people_with_boxes[person_identified].append(image_with_name)
        elif person_identified:
            self.identified_people[person_identified] = [photo]
            self.identified_people_with_boxes[person_identified] = [image_with_name]
        return person_identified

    def get_images_of_the_person_to_identify(self):
        return self.identified_people

    def get_boxes_images_of_the_person_to_identify(self):
        return self.identified_people_with_boxes

    def get_boxes_images_without_anybody(self):
        return self.photo_without_anybody
