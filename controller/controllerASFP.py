import tkinter as tk

from model.asfpModel import ASFPModel
from view.viewASFP import ViewASFP


class ControllerASFP:
    def __init__(self):
        self.root = tk.Tk()
        self.model = ASFPModel()
        self.view = ViewASFP(self.root, self.model)

    def run(self):
        self.root.mainloop()


def main():
    c = ControllerASFP()
    c.run()


if __name__ == '__main__':
    main()