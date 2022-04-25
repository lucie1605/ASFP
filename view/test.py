from tkinter import *


ws = Tk()
ws.title('PythonGuides')
ws.config(bg='#5F734C')

frame = Frame(
    ws,
    bg='#A8B9BF'
    )

can1 = Canvas(frame, width=250, height=250, bg='red')
photo1 = PhotoImage(file='../tests/images/abba.png', width=250, height=250)
item = can1.create_image(0, 0, image=photo1)
can2 = Canvas(frame, width=250, height=250, bg='red')
photo2 = PhotoImage(file='../tests/images/abba.png', width=250, height=250)
item = can2.create_image(0, 0, image=photo2)
can1.grid(row=0, column=0)
can2.grid(row=0, column=1)
sb = Scrollbar(
    frame,
    orient=VERTICAL
    )

sb.grid(row=0, column=1, sticky=NS)

can1.config(yscrollcommand=sb.set)
can2.config(yscrollcommand=sb.set)
sb.config(command=ws.yview)


ws.mainloop()