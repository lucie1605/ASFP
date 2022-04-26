import ntpath
import os
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk


class ViewASFP:
    WIDTH = 800
    HEIGHT = 600
    NB_IMAGES_PER_LINE = 5
    FORMAT_FAMILY_PHOTO = (250, 250)
    FORMAT_IMAGE_MEMBERS = (150, 150)
    INDEX_ROW_IMAGES = 15
    NB_COLUMNS = 5
    DEFAULT_MODEL_DETECTION_PARAM = 3

    def __init__(self, root, model):
        self.family_photos_path = None
        self.members_photos_path = None
        self.root = root
        self.root.title("Automatic Sort for Family Photos")
        self.root.geometry(f"{ViewASFP.WIDTH}x{ViewASFP.HEIGHT}")
        self.model = model
        self.msg_load_family = tk.StringVar()
        self.msg_load_members = tk.StringVar()
        self.members = {}

        window_buttons_load = tk.PanedWindow()
        row = 0 # to 1
        nb_rows = 3
        window_buttons_load.grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, rowspan= nb_rows)
        tk.Button(window_buttons_load, text='load family photos', command=self.load_family_photos).grid(row=0, column=0, padx=2, pady=2)
        tk.Label(window_buttons_load, textvariable=self.msg_load_family).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(window_buttons_load, text='load members photos', command=self.load_members_photos).grid(row=1, column=0, padx=2, pady=2)
        tk.Label(window_buttons_load, textvariable=self.msg_load_members).grid(row=1, column=1, padx=2, pady=2)
        tk.Button(window_buttons_load, text='apply face recognition',
                  command=self.recognize_members_in_family_photos).grid(row=2, column=0, padx=2, pady=2)

        # show all faces to recognize:
        row += nb_rows  # to 3
        # TODO nb_rows must not be constant
        nb_rows = 1 # depends on the nb of images nbImagesTotal / NB_IMAGES_PER_LINE x2
        self.window_display_members = tk.PanedWindow()
        self.window_display_members.grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, rowspan=nb_rows)

        # show all faces to recognize:
        row += nb_rows  # to end
        nb_rows = 5  # depends on the nb of images nbImagesTotal / NB_IMAGES_PER_LINE x2
        self.window_display_family_pict = tk.PanedWindow()
        self.window_display_family_pict.grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, rowspan=nb_rows)

        self.last_row = row+nb_rows
        row += nb_rows

    def load_family_photos(self):
        self.family_photos_path = filedialog.askdirectory()
        self.msg_load_family.set(f"1 folder loaded : {self.family_photos_path}")
        # TODO : put number of photos

    def load_members_photos(self):
        self.members_photos_path = filedialog.askdirectory()
        self.msg_load_members.set(f"1 folder loaded : {self.members_photos_path}")
        #TODO : put number of photos
        self.display_members_of_family()

    def display_members_of_family(self):
        if self.members_photos_path:
            row = ViewASFP.INDEX_ROW_IMAGES
            column = 0
            nb_images = 0
            for filename in os.listdir(self.members_photos_path):
                f = self.members_photos_path + "/" + filename
                # TODO : accepter seulement les fichiers image
                if os.path.isfile(f):
                    faces = self.model.detectionModel.get_detected_faces(f, ViewASFP.DEFAULT_MODEL_DETECTION_PARAM)
                    self.show_image_after_detection(faces[0],self.window_display_members,
                                                    row+1, column, ViewASFP.FORMAT_IMAGE_MEMBERS)
                    name = ntpath.basename(f).split(".")[0]
                    self.members[name] = faces[0]
                    tk.Label(self.window_display_members, text=name).grid(row=row, column=column, padx=2, pady=2)
                    column += 1
                    nb_images+=1
                    if column == ViewASFP.NB_IMAGES_PER_LINE:
                        row += 2
                        column = 0

    def show_image_after_detection(self, image, window, row=1, column=0, format=(150,150)):
        blue, green, red = cv2.split(image)
        img = cv2.merge((red, green, blue))
        img = Image.fromarray(img)
        img.thumbnail(format, Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(img)
        panel = tk.Label(window, image=imgtk)
        panel.image = imgtk
        panel.grid(row=row, column=column)

    def recognize_members_in_family_photos(self):
        if self.family_photos_path:
            nb_images = 0
            start = time.time()
            self.model.recognize_faces(self.members, self.family_photos_path)
            dico_with_boxes = self.model.get_boxes_images_of_the_person_to_identify()
            end = time.time()
            tabControl = ttk.Notebook(self.window_display_family_pict)
            for k,v in dico_with_boxes.items():
                column = 0
                #row = ViewASFP.INDEX_ROW_IMAGES
                row=0
                tab = ttk.Frame(tabControl, width= ViewASFP.WIDTH)
                tabControl.add(tab, text=k)
                tk.Label(tab, text=f"member : {k} found in {len(v)} photos").grid(row=row,column=0,padx=2, pady=2)
                for photo in v:
                    self.show_image_after_detection(photo, tab, row + 1, column, ViewASFP.FORMAT_FAMILY_PHOTO)
                    column += 1
                    nb_images += len(v)
                    if column == ViewASFP.NB_IMAGES_PER_LINE:
                        row += 1
                        column = 0
            # tabControl.pack(expand=True, fill="both")
            tabControl.grid(row=0, column=0, sticky="ew")
        tk.Label(self.root, text=f"Total : {end-start} seconds for {nb_images} images.").grid(row=self.last_row , column=0,
                                                                                                     sticky="w")

