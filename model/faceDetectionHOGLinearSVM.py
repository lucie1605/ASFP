from model.faceDetection import FaceDetection
from model.helpers import convert_and_trim_bb
import dlib


class FaceDetectionHOGLinearSVM(FaceDetection):
    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()

    def get_faces_rectangles(self, modelParam=2):
        rects = self.detector(self.formatedImage, modelParam)
        return [convert_and_trim_bb(self.formatedImage, r) for r in rects]
