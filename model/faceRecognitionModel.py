import os
import time
import dlib
import math


class FaceRecognitionModel:
    MODELS_PATH = os.path.dirname(__file__) + "\\dat_files\\"

    def __init__(self):
        self.__shape_predictor = \
            dlib.shape_predictor(FaceRecognitionModel.MODELS_PATH + "shape_predictor_5_face_landmarks.dat")
        self.__face_recognition_model = \
            dlib.face_recognition_model_v1(FaceRecognitionModel.MODELS_PATH + "dlib_face_recognition_resnet_model_v1.dat")
        self.timeElapsed = 0

    def get_identified_person(self, face, people_known):
        # face is an image of one face to identify
        # people_known is a dico with key = name, value = faceImage
        # this function should return the name (key of people_known) of the identified person
        # or none

        # Compute the vector descriptor of the face to identify
        rectangle = dlib.rectangle(left=0, top=0, right=face.shape[0], bottom=face.shape[1])
        face_to_id_shape = self.__shape_predictor(face, rectangle)
        face_to_id_descriptor = self.__face_recognition_model.compute_face_descriptor(face, face_to_id_shape)

        best_match_distance = 0.6
        best_match_name = None
        # Loop over all the known faces (people_to_identify)
        for name, known_face in people_known.items():
            # Compute the vector descriptor of the current known face
            rectangle = dlib.rectangle(left=0, top=0, right=known_face.shape[0], bottom=known_face.shape[1])
            known_face_shape = self.__shape_predictor(known_face, rectangle)
            known_face_descriptor = self.__face_recognition_model.compute_face_descriptor(known_face, known_face_shape)

            # Compute the distance between the descriptors of the face to id and the known face
            distance = math.dist(face_to_id_descriptor, known_face_descriptor)

            if distance < best_match_distance:
                # There is a better match than previous best one
                best_match_distance = distance
                best_match_name = name
        # End of loop, return the name of the best match if there is one
        return best_match_name