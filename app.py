import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from predict import load_model, load_class_indices, predict_image

class LungCancerClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Classifier")
        
        # Load model and class indices
        self.model = load_model()
        self.class_indices = load_class_indices()
        
        # Configure UI
        self.upload_btn = tk.Button(
            root, 
            text="Upload CT Scan", 
            command=self.upload_image,
            padx=10, 
            pady=5
        )
        self.upload_btn.pack(pady=20)
        
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        self.result_label = tk.Label(
            root, 
            text="", 
            font=('Helvetica', 14), 
            fg='#333333'
        )
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Display image
                img = Image.open(file_path)
                img.thumbnail((300, 300))  # Resize for display
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk  # Prevent garbage collection
                
                # Get prediction
                prediction = predict_image(file_path, self.model, self.class_indices)
                self.result_label.configure(
                    text=f"Prediction: {prediction['class']}\nConfidence: {prediction['confidence']:.2f}%"
                )
            except Exception as e:
                self.result_label.configure(text=f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerClassifierApp(root)
    root.mainloop()