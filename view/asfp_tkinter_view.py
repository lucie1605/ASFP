from tkinter import *
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
from model.faceDetectionHOGLinearSVM import FaceDetectionHOGLinearSVM
import os

picture = ""
directory_path = ""
NB_IMAGES_PER_COLUMN = 5



def show_image_after_detection(image, row= 1, column=0):
    blue, green, red = cv2.split(image)
    img = cv2.merge((red, green, blue))
    img = Image.fromarray(img)
    img = img.resize((250,250))
    imgtk = ImageTk.PhotoImage(img)
    panel = Label(root, image=imgtk)
    panel.image = imgtk
    panel.grid(row=row, column=column)

def load_img():
    global picture
    picture =  filedialog.askopenfilename(initialdir = "/",title = "Select file",
                                                filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

def load_directory():
    global directory_path
    directory_path = filedialog.askdirectory()

def apply_face_detection_on_one_image(image, row= 1, column=0):
    model = FaceDetectionHOGLinearSVM()
    model.get_detected_faces(image)
    image_with_boxes = model.get_image_with_boxes()
    show_image_after_detection(image_with_boxes, row, column)

def apply_face_detection():
    if picture:
        apply_face_detection_on_one_image(picture)
    if directory_path:
        row = 1
        column = 0
        for filename in os.listdir(directory_path):
            f = directory_path+"/"+filename
            #TODO : accepter seulement les fichiers image
            if os.path.isfile(f):
                apply_face_detection_on_one_image(f, row, column)
                column += 1
                if column == NB_IMAGES_PER_COLUMN:
                    row += 1
                    column = 0


root = Tk()
root.resizable(width=True, height=True)
Button(root, text='load image', command=load_img).grid(row=0, column=0)
Button(root, text='load directory', command=load_directory).grid(row=0, column=1)
Button(root, text='apply face detection', command=apply_face_detection).grid(row=0, column=2)

root.mainloop()