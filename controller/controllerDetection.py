import tkinter as tk
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
from view.viewDetection import ViewDetection


class ControllerASFP:
    def __init__(self):
        self.root = tk.Tk()
        self.model = FaceDetectionHOGLinearSVM()
        self.view = ViewDetection(self.root, self.model)

    def run(self):
        self.root.mainloop()


def main():
    c = ControllerASFP()
    c.run()


if __name__ == '__main__':
    main()