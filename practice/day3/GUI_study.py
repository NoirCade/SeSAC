import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageEnhance

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("이미지 밝기 조절")

        # 초기 윈도우 사이즈 조정
        self.root.geometry("400x300")

        # 이미지 로드 버튼
        self.load_button = tk.Button(root, text="이미지 열기", command=self.load_image)
        self.load_button.pack(pady=10)

        # 이미지 밝기 조절 슬라이더
        self.brightness_label = tk.Label(root, text="밝기 조절")
        self.brightness_label.pack()
        self.brightness_scale = tk.Scale(root, from_=0.0, to=2.0, orient="horizontal", resolution=0.05, command=self.update_brightness)
        self.brightness_scale.set(1.0)  # 초기 밝기 설정
        self.brightness_scale.pack(pady=10)

        # 이미지 저장 버튼
        self.save_button = tk.Button(root, text="이미지 저장", command=self.save_image)

        # 종료 버튼
        self.quit_button = tk.Button(root, text="종료", command=root.destroy)
        self.quit_button.pack(pady=10)

        # 이미지 출력 영역
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("이미지 파일", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        if hasattr(self, 'original_image'):
            self.tk_image = ImageTk.PhotoImage(self.original_image)
            self.image_label.config(image=self.tk_image, width=self.tk_image.width(), height=self.tk_image.height())

            # 윈도우 크기를 이미지에 맞게 동적으로 조절
            self.root.geometry(f"{(self.tk_image.width()+200)}x{(self.tk_image.height())+200}")
            self.brightness_scale.set(1.0)

    def update_brightness(self, value):
        if hasattr(self, 'original_image'):
            brightness_factor = float(value)
            enhanced_image = ImageEnhance.Brightness(self.original_image)
            image_with_adjusted_brightness = enhanced_image.enhance(brightness_factor)
            self.tk_image = ImageTk.PhotoImage(image_with_adjusted_brightness)
            self.image_label.config(image=self.tk_image, width=self.tk_image.width(), height=self.tk_image.height())

            # 윈도우 크기를 이미지에 맞게 동적으로 조절
            self.root.geometry(f"{(self.tk_image.width()+200)}x{(self.tk_image.height())+200}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
