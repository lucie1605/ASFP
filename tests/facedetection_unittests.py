import unittest
from model.faceDetectionCNN import FaceDetectionCNN
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM

IMAGE_PATH = "images/"
SHOW_IMAGES = False


class FaceDetectionModelsTests(unittest.TestCase):
    def test_CNN(self):
        f = FaceDetectionCNN()
        faces = f.get_detected_faces(IMAGE_PATH + "beattles_2.jpg", 0)
        if SHOW_IMAGES:
            f.show_image_with_boxes()
        self.assertEqual(len(faces), 4)  # add assertion here

    def test_HOGLinearSVM(self):
        f = FaceDetectionHOGLinearSVM()
        faces = f.get_detected_faces(IMAGE_PATH +"groupe_1.jpg", 3)
        if SHOW_IMAGES:
            f.show_image_with_boxes()
        self.assertEqual(len(faces), 18)  # add assertion here


if __name__ == '__main__':
    unittest.main()
