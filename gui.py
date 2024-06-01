import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
import pywt
from PIL import Image, ImageTk

def generate_random_watermark(size):
    watermark = np.random.randint(0, 256, size, dtype=np.uint8)
    return watermark

def pad_image_to_even(image):
    padded_image = np.pad(image, [(0, image.shape[0] % 2), (0, image.shape[1] % 2)], mode='constant')
    return padded_image

def rdwt2(image, wavelet='haar'):
    image = pad_image_to_even(image)
    coeffs = pywt.swt2(image, wavelet, level=1)
    return coeffs[0]

def irdwt2(coeffs, wavelet='haar'):
    return pywt.iswt2([coeffs], wavelet)

def embed_watermark(image, watermark):
    cA, (cH, cV, cD) = rdwt2(image)
    
    # Generate random position for watermark embedding
    x = np.random.randint(0, cA.shape[0] - watermark.shape[0])
    y = np.random.randint(0, cA.shape[1] - watermark.shape[1])

    # Embed watermark at the random position
    cA[x:x + watermark.shape[0], y:y + watermark.shape[1]] += watermark * 0.1
    
    coeffs = (cA, (cH, cV, cD))
    watermarked_image = irdwt2(coeffs)
    return np.uint8(watermarked_image), (x, y)

def extract_watermark(image, original_image, watermark_size, position):
    cA_image, _ = rdwt2(image)
    cA_original, _ = rdwt2(original_image)
    
    x, y = position
    watermark = (cA_image[x:x + watermark_size[0], y:y + watermark_size[1]] - cA_original[x:x + watermark_size[0], y:y + watermark_size[1]]) * 10
    return np.uint8(watermark)

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermarking Citra Medis")
        self.root.geometry("500x630")
        self.root.configure(bg='#000000')  # Set the background color of the root window

        # Title header
        self.header_label = tk.Label(root, text="WATERMARKING CITRA MEDIS", bg='#000000', fg='#F6C81D', font=("Fixedsys", 24))
        self.header_label.pack(pady=10)

        self.image_frame = tk.Frame(root, bd=2, relief=tk.SUNKEN, width=400, height=400, bg="#F6C81D")  # Change the background color of the frame
        self.image_frame.pack(side=tk.TOP, padx=20, pady=10)
        self.image_frame.pack_propagate(0)

        self.image_label = tk.Label(self.image_frame, bg="#009B3A")  # Change the background color of the label
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        button_width = 20
        button_height = 2

        self.load_image_button = tk.Button(root, text="Muat Gambar", command=self.load_image, bg="#B21F1F", fg="#F6C81D", width=button_width, height=button_height, bd=0, font=("Fixedsys", 16))
        self.load_image_button.pack(pady=5)

        self.embed_button = tk.Button(root, text="Embed Watermark", command=self.embed_watermark, bg="#F6C81D", fg="#296330", width=button_width, height=button_height, bd=0, font=("Fixedsys", 16))
        self.embed_button.pack(pady=5)

        self.extract_button = tk.Button(root, text="Ekstrak Watermark", command=self.extract_watermark, bg="#296330", fg="#F6C81D", width=button_width, height=button_height, bd=0, font=("Fixedsys", 16))
        self.extract_button.pack(pady=5)

        self.image = None
        self.original_image = None
        self.watermark_position = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.original_image = self.image.copy()
            self.display_image(self.image)

    def display_image(self, image):
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((500, 500), Image.Resampling.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk

    def embed_watermark(self):
        if self.image is not None:
            watermark_size = (128, 128)
            watermark = generate_random_watermark(watermark_size)
            self.image, self.watermark_position = embed_watermark(self.image, watermark)
            self.display_image(self.image)

            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.image)
                messagebox.showinfo("Sukses", "Watermark berhasil disisipkan dan disimpan")
        else:
            messagebox.showerror("Error", "Tidak ada gambar yang dimuat")

    def extract_watermark(self):
        if self.image is not None and self.original_image is not None:
            watermark_size = (128, 128)
            if self.watermark_position is not None:
                extracted_watermark = extract_watermark(self.image, self.original_image, watermark_size, self.watermark_position)
                extracted_image = Image.fromarray(extracted_watermark)

                file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
                if file_path:
                    extracted_image.save(file_path)
                    messagebox.showinfo("Watermark Extracted", "Watermark berhasil diekstrak dan disimpan")
            else:
                messagebox.showerror("Error", "Posisi watermark tidak diketahui")
        else:
            messagebox.showerror("Error", "Tidak ada gambar yang dimuat atau gambar asli hilang")

if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkApp(root)
    root.mainloop()
