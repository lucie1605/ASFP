import os
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog
from tkinter import ttk


def apply_grid(widget, row, column=0):
    widget.grid(row=row, column=column, padx=2, pady=2, sticky="w")


def apply_grid_all_columns(widget, row):
    widget.grid(row=row, column=0, columnspan=ViewASFP.NB_COLUMNS, padx=2, pady=2, sticky="w")


def create_label_title(window, text_label, row):
    label = tk.Label(window, text=text_label, font=ViewASFP.FONT_SIZE_TITLE)
    apply_grid_all_columns(label, row)


class ViewASFP:
    WIDTH = 800
    HEIGHT = 600
    NB_IMAGES_PER_LINE = 5
    FORMAT_FAMILY_PHOTO = (350, 350)
    FORMAT_IMAGE_MEMBERS = (150, 150)
    NB_COLUMNS = 5
    FONT_SIZE_TITLE = ("Arial", 13)

    def __init__(self, root, model):
        self.family_photos_path = None
        self.members_photos_path = None
        self.root = root
        self.root.title("Automatic Sort for Family Photos")
        self.root.geometry(f"{ViewASFP.WIDTH}x{ViewASFP.HEIGHT}")
        self.model = model
        self.msg_load_family = tk.StringVar()
        self.msg_load_members = tk.StringVar()

        window_buttons_load = tk.PanedWindow()
        apply_grid_all_columns(window_buttons_load, 0)
        create_label_title(window_buttons_load, "Select a directory containing all Family Photos:", 0)
        apply_grid(tk.Button(window_buttons_load, text='load family photos', command=self.load_family_photos), 1)
        apply_grid(tk.Label(window_buttons_load, textvariable=self.msg_load_family), 1, 1)
        create_label_title(window_buttons_load, "Select a directory containing photo of members to identify:", 2)
        apply_grid_all_columns(tk.Label(window_buttons_load, text="/!\\ One member per photo"), 3)
        apply_grid(tk.Button(window_buttons_load, text='load members photos', command=self.load_members_photos),4)
        apply_grid(tk.Label(window_buttons_load, textvariable=self.msg_load_members), 4, 1)

        # show all faces to recognize:
        self.window_display_members = tk.PanedWindow()
        apply_grid_all_columns(self.window_display_members, 1)

        # Recognition
        window_buttons_recognition = tk.PanedWindow()
        apply_grid_all_columns(window_buttons_recognition, 2)
        create_label_title(window_buttons_recognition, "Face Recognition:", 0)
        apply_grid(tk.Button(window_buttons_recognition, text='apply face recognition',
                  command=self.recognize_members_in_family_photos), 1)

        # show all faces to recognize:
        self.window_display_family_pict = tk.PanedWindow()
        apply_grid_all_columns(self.window_display_family_pict, 3)

        # show time elapsed:
        self.window_time_elapsed = tk.PanedWindow()
        apply_grid_all_columns(self.window_time_elapsed, 4)

    def load_family_photos(self):
        self.family_photos_path = filedialog.askdirectory()
        nb_photos = len(os.listdir(self.family_photos_path))
        self.msg_load_family.set(f"{nb_photos} photos loaded in folder : {self.family_photos_path}")

    def load_members_photos(self):
        self.members_photos_path = filedialog.askdirectory()
        nb_photos = len(os.listdir(self.members_photos_path))
        self.msg_load_members.set(f"{nb_photos} photos loaded in folder : {self.members_photos_path}")
        self.display_members_of_family()

    def display_members_of_family(self):
        if self.members_photos_path:
            create_label_title(self.window_display_members, "Family Members to Recognize:", 0)
            row = 1
            column = 0
            members = self.model.detect_members_of_family(self.members_photos_path)
            members_sorted = sorted(members)
            for name in members_sorted:
                self.show_image_after_detection(members[name], self.window_display_members,
                                                row + 1, column, ViewASFP.FORMAT_IMAGE_MEMBERS)
                tk.Label(self.window_display_members, text=name).grid(row=row, column=column, padx=2, pady=2)
                column += 1
                if column == ViewASFP.NB_IMAGES_PER_LINE:
                    row += 2
                    column = 0

    def show_image_after_detection(self, image, window, row=1, column=0, format=(150, 150)):
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
            start = time.time()
            self.model.recognize_faces(self.family_photos_path)
            dico_with_boxes = self.model.get_boxes_images_of_the_person_to_identify()
            members_sorted = sorted(dico_with_boxes)
            end = time.time()
            tabControl = ttk.Notebook(self.window_display_family_pict)
            for member in members_sorted:
                column = 0
                row = 1
                tab = ttk.Frame(tabControl, width=ViewASFP.WIDTH)
                tabControl.add(tab, text=f"{member} ({len(dico_with_boxes[member])} recognitions)")
                apply_grid(tk.Label(tab, text=f"member : {member} found in {len(dico_with_boxes[member])} photos"),row)
                for photo in dico_with_boxes[member]:
                    self.show_image_after_detection(photo, tab, row + 1, column, ViewASFP.FORMAT_FAMILY_PHOTO)
                    column += 1
                    if column == ViewASFP.NB_IMAGES_PER_LINE:
                        row += 1
                        column = 0
            row=0
            for photo in self.model.get_boxes_images_without_anybody():
                tab = ttk.Frame(tabControl, width=ViewASFP.WIDTH)
                tabControl.add(tab, text=f"Photos without any "
                                         f"recognitions({len(self.model.get_boxes_images_without_anybody())})")
                self.show_image_after_detection(photo, tab, row + 1, column, ViewASFP.FORMAT_FAMILY_PHOTO)
                column += 1
                if column == ViewASFP.NB_IMAGES_PER_LINE:
                    row += 1
                    column = 0
            tabControl.grid(row=0, column=0, sticky="w")
        apply_grid(tk.Label(self.window_time_elapsed, text=f"Total : {end - start} seconds."), 0)
