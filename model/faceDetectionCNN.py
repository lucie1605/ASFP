from model.faceDetection import FaceDetection
from model.helpers import convert_and_trim_bb
import dlib


class FaceDetectionCNN(FaceDetection):
    model_path = FaceDetection.MODELS_PATH + "mmod_human_face_detector.dat"

    def __init__(self):
        super().__init__()
        self.detector = dlib.cnn_face_detection_model_v1(FaceDetectionCNN.model_path)

    def get_faces_rectangles(self, modelParam=2):
        rects = self.detector(self.formatedImage, modelParam)
        return [convert_and_trim_bb(self.formatedImage, r.rect) for r in rects]

