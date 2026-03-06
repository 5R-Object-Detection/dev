import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, Any


def launch_gui() -> Dict[str, Any]:
    """
    Launches a clean, minimal configuration GUI.
    Returns a dictionary with all user-selected settings, or None if cancelled.
    """
    result = {"source": None, "type": "video"}

    root = tk.Tk()
    root.title("Toyota 5R: Behavioral AI System")
    root.geometry("420x400")
    root.resizable(False, False)
    root.eval('tk::PlaceWindow . center')

    # --- Source Selection ---
    tk.Label(root, text="Pilih Sumber", font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill="x", padx=15, pady=(15, 2))
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=15, pady=(0, 8))

    var_fence = tk.BooleanVar(value=True)
    var_phone = tk.BooleanVar(value=True)
    var_pocket = tk.BooleanVar(value=True)

    def save_and_close(source, run_type):
        result["source"] = source
        result["type"] = run_type
        result["enable_fence"] = var_fence.get()
        result["enable_phone"] = var_phone.get()
        result["enable_pocket"] = var_pocket.get()
        root.destroy()

    def use_webcam():
        save_and_close(0, "video")

    def use_video_file():
        file_path = filedialog.askopenfilename(
            title="Pilih File Video Lokal",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            save_and_close(file_path, "video")

    def use_image_file():
        file_path = filedialog.askopenfilename(
            title="Pilih File Gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            save_and_close(file_path, "image")

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", padx=15, pady=(0, 10))

    tk.Button(btn_frame, text="Kamera Internal", command=use_webcam).pack(side="left", expand=True, fill="x", padx=(0, 3))
    tk.Button(btn_frame, text="File Video", command=use_video_file).pack(side="left", expand=True, fill="x", padx=(0, 3))
    tk.Button(btn_frame, text="File Gambar", command=use_image_file).pack(side="left", expand=True, fill="x")

    # --- Detection Config ---
    tk.Label(root, text="Konfigurasi Deteksi", font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill="x", padx=15, pady=(10, 2))
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=15, pady=(0, 8))

    tk.Checkbutton(root, text="Virtual Fence (Area Terlarang)", variable=var_fence).pack(anchor="w", padx=15, pady=2)
    tk.Checkbutton(root, text="Phone Detection (Bermain HP)", variable=var_phone).pack(anchor="w", padx=15, pady=2)
    tk.Checkbutton(root, text="Pocket Hands Detection (Tangan di saku)", variable=var_pocket).pack(anchor="w", padx=15, pady=2)

    # --- Shortcuts ---
    tk.Label(root, text="Pintasan Keyboard", font=("TkDefaultFont", 10, "bold"), anchor="w").pack(fill="x", padx=15, pady=(15, 2))
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=15, pady=(0, 8))

    shortcuts = [
        ("R", "Gambar ulang Virtual Fence"),
        ("S", "Simpan gambar hasil (Mode Gambar)"),
        ("Q / ESC", "Keluar"),
        ("ENTER", "Konfirmasi gambar fence"),
    ]
    for key, desc in shortcuts:
        row = tk.Frame(root)
        row.pack(anchor="w", padx=15, pady=1)
        tk.Label(row, text=f"[{key}]", font=("Courier", 9, "bold"), width=10, anchor="w").pack(side="left")
        tk.Label(row, text=desc, font=("TkDefaultFont", 9), anchor="w").pack(side="left")

    root.mainloop()
    return result
