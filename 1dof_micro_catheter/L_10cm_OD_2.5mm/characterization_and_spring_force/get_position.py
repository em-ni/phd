import tkinter as tk
from PIL import Image, ImageTk

def click_event(event):
    x, y = event.x, event.y
    print("X:", x, "Y:", y)

root = tk.Tk()
img = Image.open("data\\images\\test.png")
tk_img = ImageTk.PhotoImage(img)
canvas = tk.Canvas(root, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor="nw", image=tk_img)
canvas.bind("<Button-1>", click_event)

root.mainloop()
