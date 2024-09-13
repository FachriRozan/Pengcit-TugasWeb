from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import io
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import zipfile
import matplotlib
import base64

matplotlib.use('Agg')
app = Flask(__name__)
CORS(app)
# Load pre-trained face detector model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face detection function
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Face blurring function
def blur_faces(image, faces, blur_value):
    # Ensure blur_value is a positive odd number
    blur_value = max(1, int(blur_value))  # Convert to integer and ensure it's at least 1
    if blur_value % 2 == 0:  # Ensure blur_value is odd
        blur_value += 1

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (blur_value, blur_value), 0)  # Use adjusted blur_value
        image[y:y+h, x:x+w] = blurred_face
    return image

# Edge detection function
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # Canny edge detection
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert single channel to 3 channels
    return edges_colored

# Histogram functions
def calculate_rgb_histogram(image):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("RGB Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    return img_buf

def calculate_grayscale_histogram(image):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist, color='k')
    plt.xlim([0, 256])

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    return img_buf

def equalize_image(image, mode='RGB'):
    if mode == 'RGB':
        chans = cv2.split(image)
        equalized_chans = [cv2.equalizeHist(chan) for chan in chans]
        return cv2.merge(equalized_chans)
    else:  # Grayscale
        return cv2.equalizeHist(image)


def apply_edge_detection(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'canny':
        edges = cv2.Canny(gray, 100, 200)
    elif method == 'sobel':
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5) + cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    elif method == 'prewitt':
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edges = cv2.filter2D(gray, -1, kernelx) + cv2.filter2D(gray, -1, kernely)
    elif method == 'laplacian':
        edges = cv2.Laplacian(gray, cv2.CV_64F)
    else:
        raise ValueError("Unsupported edge detection method")
    return edges
# API for face detection
@app.route('/api/face-detection', methods=['POST'])
def face_detection():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Perform face detection
    faces = detect_faces(img_cv)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert image back to PIL for sending as response
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for face blurring
@app.route('/api/face-blur', methods=['POST'])
def face_blur():
    file = request.files['image']
    blur_value = int(request.form.get('blur_value', 15))  # Get blur value from form data, default to 15 if not provided

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Perform face detection
    faces = detect_faces(img_cv)

    # Perform face blurring
    blurred_img = blur_faces(img_cv, faces, blur_value)

    # Convert image back to PIL for sending as response
    img_pil = Image.fromarray(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for edge detection
@app.route('/api/edge-detection', methods=['POST'])
def edge_detection():
    file = request.files['image']
    method = request.form.get('method', 'canny')  # Dapatkan metode dari request
    
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Terapkan metode edge detection yang dipilih
    edges = apply_edge_detection(img_cv, method)

    # Convert hasil edge detection kembali ke gambar
    edges_image = Image.fromarray(np.uint8(edges))

    # Siapkan untuk dikirim kembali sebagai respons
    img_io = BytesIO()
    edges_image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# Histogram equalization

@app.route('/rgb/histogram/original', methods=['POST'])
def rgb_original_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img_buf = calculate_rgb_histogram(img_cv)
    return send_file(img_buf, mimetype='image/png')

@app.route('/rgb/histogram/equalized', methods=['POST'])
def rgb_equalized_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    equalized_img = equalize_image(img_cv, 'RGB')
    img_buf = calculate_rgb_histogram(equalized_img)
    return send_file(img_buf, mimetype='image/png')

@app.route('/rgb/image/equalized', methods=['POST'])
def rgb_equalized_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    equalized_img = equalize_image(img_cv, 'RGB')
    img_pil = Image.fromarray(cv2.cvtColor(equalized_img, cv2.COLOR_BGR2RGB))
    
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/greyscale/histogram/original', methods=['POST'])
def grayscale_original_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')
    img_np = np.array(img)

    img_buf = calculate_grayscale_histogram(img_np)
    return send_file(img_buf, mimetype='image/png')

@app.route('/greyscale/histogram/equalized', methods=['POST'])
def grayscale_equalized_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')
    img_np = np.array(img)

    equalized_img = equalize_image(img_np, 'Grayscale')
    img_buf = calculate_grayscale_histogram(equalized_img)
    return send_file(img_buf, mimetype='image/png')

@app.route('/greyscale/image/equalized', methods=['POST'])
def grayscale_equalized_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')
    img_np = np.array(img)

    equalized_img = equalize_image(img_np, 'Grayscale')
    img_pil = Image.fromarray(equalized_img)
    
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')
@app.route('/greyscale/image/original', methods=['POST'])
def grayscale_original_image():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')  # Mengubah gambar ke mode grayscale
    img_np = np.array(img)
    
    # Simpan gambar grayscale ke dalam BytesIO buffer
    img_io = BytesIO()
    img_pil = Image.fromarray(img_np)
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
