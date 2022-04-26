import os
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
import time

class ViewASFP:
    NB_IMAGES_PER_COLUMN = 5
    FORMAT_IMAGE_OUTPUT = (250,250)
    INDEX_ROW_IMAGES = 3

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.model_param = tk.IntVar()
        self.directory_path = None
        self.picture = None
        self.info = None

        tk.Button(root, text='load image', command=self.load_img).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(root, text='load directory', command=self.load_directory).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(root, text='apply face detection', command=self.apply_face_detection).grid(row=1, column=0, padx=2, pady=2)
        tk.Entry(self.root, textvariable=self.model_param).grid(row=1, column=1, padx=2, pady=2)
        tk.Label(self.root, text="input parameter model, enter a value between 0 and 5.").grid(row=1, column=2, padx=2, pady=2)


    def load_img(self):
        self.picture = filedialog.askopenfilename(initialdir="/", title="Select file",
                                             filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        tk.Label(self.root, text=f"1 image loaded : {self.picture}").grid(row=0, column=2, padx=2,
                                                                                               pady=2)

    def load_directory(self):
        self.directory_path = filedialog.askdirectory()
        tk.Label(self.root, text=f"1 folder loaded : {self.directory_path}").grid(row=0, column=2, padx=2,
                                                                             pady=2)

    def show_image_after_detection(self, image, row=1, column=0):
        blue, green, red = cv2.split(image)
        img = cv2.merge((red, green, blue))
        img = Image.fromarray(img)
        img = img.resize(ViewASFP.FORMAT_IMAGE_OUTPUT)
        imgtk = ImageTk.PhotoImage(img)
        panel = tk.Label(self.root, image=imgtk)
        panel.image = imgtk
        panel.grid(row=row, column=column)

    def apply_face_detection_on_one_image(self, image, row, column):
        self.model.get_detected_faces(imagePath=image, modelParam=self.model_param.get())
        image_with_boxes = self.model.get_image_with_boxes()
        self.show_image_after_detection(image_with_boxes, row, column)

    def apply_face_detection(self):
        if self.picture:
            self.apply_face_detection_on_one_image(self.picture, ViewASFP.INDEX_ROW_IMAGES, 0)
            self.info = self.model.get_info_about_computing()
            tk.Label(self.root, text=self.info).grid(row=ViewASFP.INDEX_ROW_IMAGES + 1, column=0, padx=2, pady=2)
        if self.directory_path:
            row = ViewASFP.INDEX_ROW_IMAGES
            column = 0
            nb_images = 0
            start = time.time()
            for filename in os.listdir(self.directory_path):
                f = self.directory_path + "/" + filename
                # TODO : accepter seulement les fichiers image
                if os.path.isfile(f):
                    self.apply_face_detection_on_one_image(f, row, column)
                    self.info = self.model.get_info_about_computing()
                    tk.Label(self.root, text=self.info).grid(row=row + 1, column=column, padx=2, pady=2)
                    column += 1
                    nb_images+=1
                    if column == ViewASFP.NB_IMAGES_PER_COLUMN:
                        row += 2
                        column = 0
            end = time.time()
            tk.Label(self.root, text=f"Total : {end-start} seconds for {nb_images} images.").grid(row=row + 2, column=0, padx=2, pady=2)
