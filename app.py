import tkinter as tk
import numpy as np

from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model

# ---------------------------------
# Load Trained Model
# ---------------------------------
model = load_model("model/digit_model.h5")

# ---------------------------------
# Main Window
# ---------------------------------
root = tk.Tk()
root.title("Handwritten Digit Recognition v2.0")
root.geometry("500x650")
root.resizable(False, False)

# ---------------------------------
# Heading
# ---------------------------------
title = tk.Label(
    root,
    text="Handwritten Digit Recognition",
    font=("Arial", 20, "bold")
)

title.pack(pady=15)

# ---------------------------------
# Canvas
# ---------------------------------
CANVAS_SIZE = 350

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    cursor="cross"
)

canvas.pack()

# ---------------------------------
# Hidden PIL Image
# ---------------------------------
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
draw_image = ImageDraw.Draw(image)

# ---------------------------------
# Variables
# ---------------------------------
last_x = None
last_y = None

BRUSH_SIZE =10

# ---------------------------------
# Mouse Press
# ---------------------------------
def start_draw(event):
    global last_x, last_y

    last_x = event.x
    last_y = event.y

# ---------------------------------
# Mouse Move
# ---------------------------------
def draw(event):

    global last_x, last_y

    x, y = event.x, event.y

    if last_x is not None and last_y is not None:

        canvas.create_line(
            last_x,
            last_y,
            x,
            y,
            width=BRUSH_SIZE,
            fill="black",
            capstyle=tk.ROUND,
            smooth=True
        )

        draw_image.line(
            [last_x, last_y, x, y],
            fill="black",
            width=BRUSH_SIZE
        )

    last_x = x
    last_y = y

# ---------------------------------
# Mouse Release
# ---------------------------------
def stop_draw(event):
    global last_x, last_y

    last_x = None
    last_y = None

# ---------------------------------
# Clear Canvas
# ---------------------------------
def clear_canvas():

    global image, draw_image

    canvas.delete("all")

    image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
    draw_image = ImageDraw.Draw(image)

    prediction_label.config(text="Prediction: --")

# ---------------------------------
# Predict (Placeholder)
# ---------------------------------
def predict_digit():

    # Convert white background to black and digit to white
    img = ImageOps.invert(image)

    # Find the handwritten digit
    bbox = img.getbbox()

    if bbox is None:
        prediction_label.config(text="Please draw a digit!")
        return

    # Crop only the digit
    img = img.crop(bbox)

    # Resize while keeping aspect ratio
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Create a blank 28x28 image
    final_img = Image.new("L", (28, 28), 0)

    # Center the digit
    x = (28 - img.width) // 2
    y = (28 - img.height) // 2

    final_img.paste(img, (x, y))

    # Convert to NumPy array
    img_array = np.array(final_img).astype("float32")

    # Normalize
    img_array = img_array / 255.0

    # Flatten
    img_array = img_array.reshape(1, 784)

    # Predict
    prediction = model.predict(img_array, verbose=0)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    prediction_label.config(
        text=f"Prediction : {predicted_digit}   ({confidence:.2f}%)"
    )

# ---------------------------------
# Bind Mouse Events
# ---------------------------------
canvas.bind("<Button-1>", start_draw)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_draw)

# ---------------------------------
# Buttons
# ---------------------------------
button_frame = tk.Frame(root)
button_frame.pack(pady=20)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    width=12,
    font=("Arial", 14),
    command=predict_digit
)

predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    width=12,
    font=("Arial", 14),
    command=clear_canvas
)

clear_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(
    button_frame,
    text="Exit",
    width=12,
    font=("Arial", 14),
    command=root.destroy
)

exit_button.grid(row=0, column=2, padx=10)

# ---------------------------------
# Prediction Label
# ---------------------------------
prediction_label = tk.Label(
    root,
    text="Prediction: --",
    font=("Arial", 18, "bold")
)

prediction_label.pack(pady=15)

root.mainloop()