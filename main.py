import tkinter
import tkinter as tk
from model import model
from tkinter import messagebox

import numpy as np
from PIL import ImageGrab

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Reconhecimento de dgitos manuscritos")
        self.canvas = tkinter.Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button_predict = tk.Button(master, text="Prever texto", command=self.predict_number)

    def predict_number(self):
        x = ImageGrab.grab(bbox=(self.canvas.winfo_x()+self.master.winfo_x(),
                                 self.canvas.winfo_y()+self.master.winfo_y(),
                                 self.master.winfo_x()+280, self.master.winfo_y()+280))
        x = x.convert("L")
        x = np.array(x)/255
        x = np.expand_dims(x, axis=0)
        prediction = model.predict(x)
        digit = np.argmax(prediction)
        messagebox.showinfo(f"Resultado", f"Digito previsto {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()