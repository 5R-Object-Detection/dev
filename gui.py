import tkinter as tk
from tkinter import filedialog
from typing import Union

def launch_gui() -> Union[str, int, None]:
    source = None
    root = tk.Tk()
    root.title("Toyota 5R System Config")
    root.geometry("400x200")
    
    # Enable grabbing focus on macOS
    root.eval('tk::PlaceWindow . center')
    
    def use_webcam():
        nonlocal source
        source = 1
        root.destroy()
        
    def use_file():
        nonlocal source
        file_path = filedialog.askopenfilename(
            title="Pilih File Video Lokal", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            source = file_path
            root.destroy()
            
    tk.Label(root, text="Behavioral Inference System", font=("Helvetica", 16, "bold")).pack(pady=15)
    
    btn_cam = tk.Button(root, text="Gunakan Kamera Internal", command=use_webcam, width=30, height=2)
    btn_cam.pack(pady=5)
    
    btn_file = tk.Button(root, text="Pilih File Video Lokal", command=use_file, width=30, height=2)
    btn_file.pack(pady=5)
    
    root.mainloop()
    return source
