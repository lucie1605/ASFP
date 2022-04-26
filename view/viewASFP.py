import os
import tkinter as tk
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
import time
from tkinter import ttk


class ViewASFP:
    NB_IMAGES_PER_COLUMN = 5
    FORMAT_IMAGE_OUTPUT = (250,250)
    INDEX_ROW_IMAGES = 3
    NB_COLUMNS = 3

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.model_param = tk.IntVar()
        self.label_family_photo_loaded = tk.StringVar()
        self.label_persons_to_identify_loaded = tk.StringVar()
        self.family_photos_directory_path = None
        self.family_picture = None
        self.info = None
        self.persons_to_id_photos_directory_path = None
        self.persons_to_id_picture = None

        # Load family photos
        # takes 2 rows
        self.load_photo_view('Load Family Photos', 0, self.label_family_photo_loaded)
        # Load photos of persons to identify
        # takes 2 rows
        self.load_photo_view('Load Photos of the Persons To Identify ', 2, self.label_persons_to_identify_loaded)
        # display photos of the persons to identify :
        row = 4
        window_person_to_identify = tk.PanedWindow(self.root)
        window_person_to_identify.grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, sticky=tk.W)


        tk.Button(root, text='apply face detection', command=self.apply_face_detection).grid(row=1, column=0, padx=2, pady=2)
        tk.Entry(self.root, textvariable=self.model_param).grid(row=1, column=1, padx=2, pady=2)
        tk.Label(self.root, text="input parameter model, enter a value between 0 and 5.").grid(row=1, column=2, padx=2, pady=2)

        # pour afficher les photos par personne identifiée
        window_resultats = tk.PanedWindow(self.root)
        window_resultats.grid(row=2, column=0, columnspan=ViewASFP.NB_COLUMNS, sticky=tk.W)
        tabControl = ttk.Notebook(window_resultats)
        self.tabs=[]
        for i in range(2):
            tab = ttk.Frame(tabControl)
            tabControl.add(tab, text=f'Tab {i}')
            self.tabs.append(tab)
        tabControl.pack(expand=1, fill="both")

    def load_photo_view(self, title, row, label_image, load_directory_cmd):
        tk.Label(self.root, text=title).grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, padx=2,
                                                            pady=2)
        row += 1
        tk.Button(self.root, text='load 1 directory', command=load_directory_cmd).grid(row=row, column=1, padx=2, pady=2)
        self.label_image_loaded.set(f"1 folder loaded : {self.family_photos_directory_path}")
        tk.Label(self.root, textvariable=label_image).grid(row=row, column=2, padx=2, pady=2)

    def load_directory_family(self):
        self.family_photos_directory_path = filedialog.askdirectory()

    def load_directory_person(self):
        self.persons_to_id_photos_directory_path = filedialog.askdirectory()

    def display_persons_to_identify_pictures(self):
        if self.family_picture:
            self.apply_face_detection_on_one_image(self.family_picture, ViewASFP.INDEX_ROW_IMAGES, 0)
            self.info = self.model.get_info_about_computing()
            tk.Label(self.root, text=self.info).grid(row=ViewASFP.INDEX_ROW_IMAGES + 1, column=0, padx=2, pady=2)
        if self.family_photos_directory_path:
            row = ViewASFP.INDEX_ROW_IMAGES
            column = 0
            nb_images = 0
            start = time.time()
            for filename in os.listdir(self.family_photos_directory_path):
                f = self.family_photos_directory_path + "/" + filename
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


    def show_image(self, image_path, widget_name, row, column):
        imgtk = Image.open(image_path)
        imgtk = ImageTk.PhotoImage(imgtk)
        panel = tk.Label(widget_name, image=imgtk)
        panel.image = imgtk
        panel.grid(row=row, column=column)


    def show_image_after_detection(self, image, row=1, column=0):
        blue, green, red = cv2.split(image)
        img = cv2.merge((red, green, blue))
        img = Image.fromarray(img)
        img.thumbnail(ViewASFP.FORMAT_IMAGE_OUTPUT, Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(img)
        panel = tk.Label(self.root, image=imgtk)
        panel.image = imgtk
        panel.grid(row=row, column=column)

    def apply_face_detection_on_one_image(self, image, row, column):
        self.model.get_detected_faces(imagePath=image, modelParam=self.model_param.get())
        image_with_boxes = self.model.get_image_with_boxes()
        self.show_image_after_detection(image_with_boxes, row, column)

    def apply_face_detection(self):
        if self.family_picture:
            self.apply_face_detection_on_one_image(self.family_picture, ViewASFP.INDEX_ROW_IMAGES, 0)
            self.info = self.model.get_info_about_computing()
            tk.Label(self.root, text=self.info).grid(row=ViewASFP.INDEX_ROW_IMAGES + 1, column=0, padx=2, pady=2)
        if self.family_photos_directory_path:
            row = ViewASFP.INDEX_ROW_IMAGES
            column = 0
            nb_images = 0
            start = time.time()
            for filename in os.listdir(self.family_photos_directory_path):
                f = self.family_photos_directory_path + "/" + filename
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
