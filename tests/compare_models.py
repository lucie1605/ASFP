import time
from model.faceDetectionCNN import FaceDetectionCNN
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
import numpy as np
import pandas as pd
from datetime import datetime

IMAGE_PATH = "images/"
RESULTS_PATH = "results/"
LOG_FILE = RESULTS_PATH+"results_log.txt"
SAVE_RESULTS = False

def print_time(phi):
    def _internal(*args, **kwargs):
        if not SAVE_RESULTS :
            return phi(*args, **kwargs)

        with open(LOG_FILE,"a") as f:
            today = datetime.today()
            start = time.time()
            f.write(f"[INFO] {today}\n")
            f.write(f"[INFO] running {phi.__name__}, {args}, {kwargs}...\n")
            calcul = phi(*args, **kwargs)
            end = time.time()
            for line in calcul.values:
                f.write(f"[INFO] Max faces detected : {max(line[:-1])}, in real there are {line[-1]} faces \n")
            msg = f"[INFO] Face detection took {end - start:.4f} seconds for {nbImages} images.\n"
            f.write(msg)
            print(msg)
        return calcul
    return _internal


@print_time
def test_faceDetection_vs_format(modelParam=0, cls=FaceDetectionHOGLinearSVM, suffixCsv=""):
    images = {"abba.png": 4, "beattles_1.jpg": 4, "beattles_2.jpg": 4, "groupe_1.jpg": 30, "groupe_2.jpg": 40}
    formatCombinations = [(True, True), (False, False), (True, False), (False, True)]
    global nbImages
    nbImages = len(images)*len(formatCombinations)
    listImages = list(images.keys())
    mat = np.zeros((5, 5))
    for j in range(4):
        for i in range(0, 5):
            imagePath = IMAGE_PATH+listImages[i]
            formatCombination = formatCombinations[j]
            f = cls()
            faces = f.get_detected_faces(imagePath, modelParam,
                                         convertToGray=formatCombination[0], resize=formatCombination[1])
            mat[i, j] = len(faces)
    # put results in last column
    mat[:, len(formatCombinations)] = list(images.values())
    df = pd.DataFrame(data=mat, columns=formatCombinations + ["Solution"], index=listImages)
    if SAVE_RESULTS:
        df.to_csv(f"{RESULTS_PATH}results_{cls.__name__}_{modelParam}{suffixCsv}.csv", "\t")
    print(df)
    return df


if __name__ == '__main__':
    # for i in range(4):
    #     test_faceDetection_vs_format(i, FaceDetectionHOGLinearSVM)
    for i in range(4):
        test_faceDetection_vs_format(i, FaceDetectionCNN)

    # 1/ tester avec ou sans formatage
    # 2/ tester en faisant varier modelParam 0 -> 3
    # 3/ comparer avec d'autres modeles
    # comparer les temps d'execution et le nb de faces détecter sur plusieurs images
