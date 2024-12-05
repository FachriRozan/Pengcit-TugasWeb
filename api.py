from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
from torchvision import models
from PIL import Image
import numpy as np
import io
import torchvision.transforms as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
from werkzeug.utils import secure_filename
import os
import logging
import base64
from skimage.util import random_noise
from skimage import util as noise
from scipy import ndimage
import huffman
import sys
import ailia
# import original modules
sys.path.append('util/')
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from detector_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_MOBILENETV2_PATH = 'mobilenetv2.onnx'
MODEL_MOBILENETV2_PATH = 'mobilenetv2.onnx.prototxt'
WEIGHT_RESNET50_PATH = 'resnet50.onnx'
MODEL_RESNET50_PATH = 'resnet50.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/background_matting_v2/'


# Inisialisasi Flask app
matplotlib.use('Agg')
app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})
# Izinkan semua asal untuk semua endpoint
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model DeepLabV3 yang sudah dilatih
SegModel = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
dic_model = {
    'mobilenetv2': (WEIGHT_MOBILENETV2_PATH, MODEL_MOBILENETV2_PATH),
    'resnet50': (WEIGHT_RESNET50_PATH, MODEL_RESNET50_PATH),
}

# Load model once when the server starts
model_type = 'mobilenetv2'  # Default model type
weight_path, model_path = dic_model[model_type]
check_and_download_models(weight_path, model_path, REMOTE_PATH)
net = ailia.Net(model_path, weight_path)

# ======================
# Main functions
# ======================

def bgr_image_from_file(file):
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("Invalid background image format")

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return preprocess(img)

def preprocess(img, shape=None):
    if shape:
        h, w = shape
        img = cv2.resize(img, (w, h))
    img = img / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    return img

def post_process(*args, a=False):
    pha, fgr, _, _, _, _ = args
    if a:
        com = np.concatenate([fgr * np.not_equal(pha, 0), pha], axis=1)
    else:
        bg_clr = np.array([120 / 255, 255 / 255, 155 / 255]).reshape((1, 3, 1, 1))
        com = pha * fgr + (1 - pha) * bg_clr
    img = com.transpose((0, 2, 3, 1))[0] * 255
    img = img.astype(np.uint8)
    return img

def predict(net, img, bgr_img):
    _, _, h, w = bgr_img.shape
    im_h, im_w = img.shape[:2]
    shape = (h, w) if im_h != h or im_w != w else None
    img = preprocess(img, shape)
    output = net.predict([img, bgr_img])
    return output
# Fungsi untuk mengaplikasikan filter pada gambar
def apply_filter(image, filter_type):
    kernel = np.ones((9,9),np.uint8)
    if filter_type == "dilation":
        return cv2.dilate(image, kernel, iterations = 1)
    elif filter_type == "erotion":
        return cv2.erode(image, kernel, iterations = 1)
    elif filter_type == "opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif filter_type == "closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Filter tidak valid")

# Image inversion function
def invert_image(image):
    inverted_image = 255 - image
    return inverted_image

# Image Dithering
def dithering(image):
    """Apply dithering to an RGB image."""
    # Convert image to numpy array
    img = np.array(image)
    h, w, _ = img.shape

    # Convert to grayscale for dithering
    gray_img = img.mean(axis=2)

    # Floyd-Steinberg dithering algorithm
    for y in range(h):
        for x in range(w):
            old_pixel = gray_img[y][x]
            new_pixel = 255 * (old_pixel > 127)  # Thresholding
            gray_img[y][x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            if x < w - 1:  # Error diffusion to the right pixel
                gray_img[y][x + 1] += quant_error * 7 / 16
            if x > 0 and y < h - 1:  # Error diffusion to the bottom-left pixel
                gray_img[y + 1][x - 1] += quant_error * 3 / 16
            if y < h - 1:  # Error diffusion to the bottom pixel
                gray_img[y + 1][x] += quant_error * 5 / 16
            if x < w - 1 and y < h - 1:  # Error diffusion to the bottom-right pixel
                gray_img[y + 1][x + 1] += quant_error * 1 / 16

    # Create a new image from the dithered grayscale
    dithered_img = Image.fromarray(np.uint8(gray_img))

    return dithered_img

# Miniature Effect 
def apply_miniature_effect(image):
    # Pastikan image adalah dalam format BGR untuk OpenCV
    # OpenCV expects BGR, while PIL is RGB
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Definisikan area fokus (tengah gambar)
    height, width = img_bgr.shape[:2]
    focus_start = int(height * 0.3)
    focus_end = int(height * 0.7)

    # Buat mask untuk area fokus
    mask = np.zeros_like(img_bgr, dtype=np.uint8)
    mask[focus_start:focus_end, :] = img_bgr[focus_start:focus_end, :]

    # Terapkan Gaussian blur ke seluruh gambar
    blurred_image = cv2.GaussianBlur(img_bgr, (15, 15), 0)

    # Gabungkan area fokus dengan latar belakang blur
    result = blurred_image
    result[focus_start:focus_end, :] = mask[focus_start:focus_end, :]

    # Tingkatkan saturasi untuk efek 'miniatur'
    img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = cv2.add(img_hsv[:, :, 1], 50)  # Tingkatkan saturasi
    final_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # Pastikan kembali ke BGR untuk penyimpanan

    # Kembalikan ke format RGB untuk digunakan lebih lanjut
    return cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

# Fungsi untuk decode hasil segmentasi menjadi gambar RGB
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Fungsi untuk melakukan segmentasi gambar
def segment_image(image):
    # Transformasi gambar sesuai model DeepLabV3
    trf = T.Compose([T.Resize(128),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Proses gambar
    inp = trf(image).unsqueeze(0)
    
    # Prediksi dengan model DeepLabV3
    with torch.no_grad():
        out = SegModel(inp)['out']
    
    # Dapatkan segmen prediksi
    predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    
    # Decode segmen prediksi menjadi gambar RGB
    rgb_image = decode_segmap(predicted)
    
    return rgb_image

# Face detection function
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Face blurring function
def blur_faces(image, faces, blur_value):
    blur_value = max(1, int(blur_value))
    if blur_value % 2 == 0:
        blur_value += 1

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (blur_value, blur_value), 0)  # Use adjusted blur_value
        image[y:y+h, x:x+w] = blurred_face
    return image

# Edge detection function
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

# Histogram functions

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
    if mode == 'RGB':
        chans = cv2.split(image)
        equalized_chans = [cv2.equalizeHist(chan) for chan in chans]
        return cv2.merge(equalized_chans)
    else:  # Grayscale
        return cv2.equalizeHist(image)

#edge detection

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
import random
import numpy as np

def add_salt_pepper_noise(image):
    row, col = image.shape[:2]
    number_of_pixels = 500  # Fixed amount of noise
    for _ in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        image[y_coord][x_coord] = 255 if random.random() > 0.5 else 0
    return image

def add_gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def add_speckle_noise(image):
    noise = np.random.randn(*image.shape) * 0.1
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def add_periodic_noise(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape[:2]
    Y = np.tile(np.arange(rows), (cols, 1)).T
    sinusoidal_noise = np.sin(2 * np.pi * Y / 30) * 30
    noisy_image = np.clip(image.astype(np.float32) + sinusoidal_noise, 0, 255).astype(np.uint8)
    return noisy_image



def scale_image(image, method, scale_percent):
    """Scale image using different interpolation methods and scale percent."""
    height, width = image.shape[:2]
    
    # Calculate new dimensions based on scale percent
    scale_factor = 1 + (scale_percent / 100)  # Scale factor based on input percent
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    if method == 'nearest':
        result = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    elif method == 'linear':
        # Perform linear interpolation in one dimension first (e.g., width)
        temp = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)
        # Then resize the other dimension (height) using the same interpolation
        result = cv2.resize(temp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'bilinear':
        # Bilinear interpolation, resizing in both directions simultaneously
        result = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'cubic':
        result = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Unsupported scaling method")

    return result

def noise_reduction(image, method='lowpass'):
    try:
        if method == 'lowpass':
            result = ndimage.uniform_filter(image, 5)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        elif method == 'rankorder':
            cross=np.array([[0,1,0],[1,1,1],[0,1,0]])
            result = ndimage.median_filter(image, footprint=cross)
            return result
        elif method == 'outlier':
                # Kernel rata-rata (average filter) berbentuk cross
                kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]) / 5.0  # Normalisasi agar jumlah kernel menjadi 1

                # Convolve gambar dengan kernel rata-rata
                image_filtered = ndimage.convolve(image.astype(np.float32), kernel)

                # Threshold untuk mendeteksi outlier (bisa disesuaikan)
                threshold = 30.0

                # Mask untuk mendeteksi piksel yang merupakan outlier
                outlier_mask = np.abs(image - image_filtered) > threshold

                # Mengganti piksel outlier dengan hasil filtering
                result = np.where(outlier_mask, image_filtered, image)

                return np.clip(result, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Invalid noise reduction method")
    except Exception as e:
        logging.error(f"Noise reduction error: {e}")
        return image



def chain_code(image):
        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply binary threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize result dictionary
            result = {
                'total_contours': len(contours),
                'chains': []
            }
            
            # Process each contour
            for idx, contour in enumerate(contours):
                chain = []
                # Get chain code for the contour
                for i in range(1, len(contour)):
                    prev = contour[i-1][0]
                    curr = contour[i][0]
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    
                    # Convert direction to chain code (0-7)
                    if dx == 1 and dy == 0: direction = 0
                    elif dx == 1 and dy == -1: direction = 1
                    elif dx == 0 and dy == -1: direction = 2
                    elif dx == -1 and dy == -1: direction = 3
                    elif dx == -1 and dy == 0: direction = 4
                    elif dx == -1 and dy == 1: direction = 5
                    elif dx == 0 and dy == 1: direction = 6
                    elif dx == 1 and dy == 1: direction = 7
                    
                    chain.append(direction)
                
                # Add chain code information for this contour
                contour_info = {
                    'contour_id': idx + 1,
                    'chain_code': chain,
                    'length': len(chain),
                    'start_point': tuple(contour[0][0].tolist())  # Starting coordinate
                }
                result['chains'].append(contour_info)
            
            return result
        except Exception as e:
            logging.error(f"Chain code error: {e}")
            return {
                'total_contours': 0,
                'chains': [],
                'error': str(e)
            }

# API for image segmentation
@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        if not file:
            return jsonify({"error": "No file provided"}), 400
        
        img = Image.open(file.stream)

        # Validasi format gambar
        if img.format not in ['JPEG', 'PNG']:
            return jsonify({"error": "Unsupported image format"}), 400

        # Proses segmentasi gambar
        segmented_image = segment_image(img)
        segmented_pil = Image.fromarray(segmented_image.astype('uint8'))

        # Simpan ke buffer untuk dikirim sebagai response
        buf = io.BytesIO()
        segmented_pil.save(buf, format='JPEG')
        buf.seek(0)

        # Kirim gambar sebagai response
        return send_file(buf, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Kirim pesan kesalahan dalam response

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
@app.route('/api/face-effect', methods=['POST'])
def face_effect():
    file = request.files['image']
    effect_value = float(request.form.get('effect_value', 50))  # Get effect value from form data, default to 50

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Perform face detection
    faces = detect_faces(img_cv)
    # Apply blur effect
    blurred_image = blur_faces(img_cv, faces, effect_value)

    # Convert image back to PIL for sending as response
    img_pil = Image.fromarray(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for edge detection
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
    
#Adither
@app.route('/api/dither', methods=['POST'])
def dither():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    # Apply dithering
    dithered_img = dithering(img)

    # Convert result back to buffer for response
    img_io = BytesIO()
    dithered_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for image inversion
@app.route('/api/image-inversion', methods=['POST'])
def image_inversion():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Perform image inversion
    inverted_image = invert_image(img_np)

    # Convert inverted image to PIL for sending as response
    inverted_img = Image.fromarray(inverted_image)
    img_io = BytesIO()
    inverted_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for Miniature Effect
@app.route("/api/miniature-effect", methods=['POST'])
def index():
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("RGB")  # Pastikan kita menggunakan format RGB
        img_cv = np.array(img)

        # Terapkan efek miniatur
        miniature_img = apply_miniature_effect(img_cv)

        # Konversi ke Image untuk respons Flask
        result_image = Image.fromarray(miniature_img)
        byte_io = BytesIO()
        result_image.save(byte_io, 'PNG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype="image/png")

    return render_template("index.html")

# Endpoint untuk morphology
@app.route('/morphology', methods=['POST'])
def apply_filter_endpoint():
    if 'image' not in request.files or 'filter' not in request.form:
        return jsonify({'error': 'Gambar atau filter tidak ditemukan'}), 400

    # Mengambil gambar dari request
    image_file = request.files['image']
    filter_type = request.form['filter'].lower()

    if filter_type not in ['dilation', 'erotion', 'opening', 'closing']:
        return jsonify({'error': 'Filter tidak valid. Pilihan filter: dilation, erosion, opening, closing'}), 400

    # Membaca gambar langsung dari stream
    image_stream = image_file.read()
    image = cv2.imdecode(np.frombuffer(image_stream, np.uint8), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    # Menerapkan filter
    filtered_image = apply_filter(image, filter_type)

    # Mengonversi hasil filter ke dalam bentuk PNG dalam memori
    _, buffer = cv2.imencode('.png', filtered_image)
    image_stream = io.BytesIO(buffer)

    # Mengirimkan gambar hasil filter ke frontend
    return send_file(image_stream, mimetype='image/png')

@app.route('/api/scale', methods=['POST'])
def scale():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    method = request.form.get('method', 'linear')
    
    # Get scale percent with a default value of 0
    try:
        scale_percent = float(request.form.get('scale_percent', '0'))
    except ValueError:
        return jsonify({'error': 'Scale percent must be a number'}), 400

    # Validate scale percent range
    if scale_percent < -100 or scale_percent > 100:
        return jsonify({'error': 'Scale percent must be between -100 and 100'}), 400

    # Read the image
    img = Image.open(file.stream)
    img_np = np.array(img)
    
    # Convert RGB to BGR for OpenCV
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Scale image using the specified method and scale percent
    scaled = scale_image(img_np, method, scale_percent)
    
    # Convert back to RGB
    if len(scaled.shape) == 3:
        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image and save to buffer
    result_image = Image.fromarray(scaled)
    img_io = BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

@app.route('/api/restore', methods=['POST'])
def restore_image():
    """Restore image by reducing noise"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        method = request.form.get('method', 'lowpass')
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if method == 'lowpass':
            restored_image = ndimage.uniform_filter(image, 5)
        elif method == 'median':
            restored_image = cv2.medianBlur(image, 5)
        elif method == 'rankorder':
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Konversi ke grayscale
            cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # Footprint berbentuk cross
            restored_image = ndimage.median_filter(gray_image, footprint=cross)
            restored_image = cv2.cvtColor(restored_image, cv2.COLOR_GRAY2BGR)  # Kembalikan ke format 3 channel
        elif method == 'outlier':
            # Kernel rata-rata (average filter) berbentuk cross
            kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]]) / 5.0  

            # Pisahkan gambar menjadi tiga channel (BGR)
            channels = cv2.split(image)

            # Proses setiap channel secara terpisah
            filtered_channels = []
            for channel in channels:
                # Convolve channel dengan kernel rata-rata
                filtered_channel = ndimage.convolve(channel.astype(np.float32), kernel)

                # Threshold untuk mendeteksi outlier (bisa disesuaikan)
                threshold = 30.0

                # Mask untuk mendeteksi piksel yang merupakan outlier
                outlier_mask = np.abs(channel - filtered_channel) > threshold

                # Mengganti piksel outlier dengan hasil filtering
                result = np.where(outlier_mask, filtered_channel, channel)
                filtered_channels.append(np.clip(result, 0, 255).astype(np.uint8))

            # Gabungkan kembali channel yang sudah diproses
            restored_image = cv2.merge(filtered_channels)

        else:
            return jsonify({"error": "Invalid restoration method"}), 400

        restored_pil = Image.fromarray(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
        img_io = BytesIO()
        restored_pil.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logging.error(f"Image restoration error: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/add_noise', methods=['POST'])
def add_noise():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']
        noise_method = request.form.get('noise_method', 'saltpepper')
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if noise_method == 'saltpepper':
            noise_img = add_salt_pepper_noise(image)
        elif noise_method == 'gaussian':
            noise_img = add_gaussian_noise(image)
        elif noise_method == 'speckle':
            noise_img = add_speckle_noise(image)
        elif noise_method == 'periodic':
            noise_img = add_periodic_noise(image)
        else:
            return jsonify({"error": "Invalid noise method"}), 400

        img_io = BytesIO()
        Image.fromarray(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)).save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logging.error(f"Noise addition error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/shape', methods=['POST'])
def extract_shape():
    """Extract chain code from image contours"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert image to grayscale and get binary image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
        
        # Extract chain codes
        shapes = []
        for idx, contour in enumerate(contours):
            chain = []
            for i in range(1, len(contour)):
                prev = contour[i - 1][0]
                curr = contour[i][0]
                dx = int(curr[0] - prev[0])  # Convert to Python int
                dy = int(curr[1] - prev[1])  # Convert to Python int
                
                # Chain code directions
                if dx == 1 and dy == 0: direction = 0
                elif dx == 1 and dy == -1: direction = 1
                elif dx == 0 and dy == -1: direction = 2
                elif dx == -1 and dy == -1: direction = 3
                elif dx == -1 and dy == 0: direction = 4
                elif dx == -1 and dy == 1: direction = 5
                elif dx == 0 and dy == 1: direction = 6
                elif dx == 1 and dy == 1: direction = 7
                else: direction = -1  # Invalid movement
                
                chain.append(direction)
            
            shapes.append({
                "shape_id": idx + 1,
                "chain_code": chain,
                "length": len(chain),
                "start_point": [int(contour[0][0][0]), int(contour[0][0][1])]  # Convert to Python list
            })
        
        # Convert output image to base64
        _, buffer = cv2.imencode('.png', output_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result = {
            "status": "success",
            "total_shapes": len(contours),
            "shapes": shapes,
            "visualization": image_base64
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        logging.error(f"Shape extraction error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
        
@app.route('/api/compress', methods=['POST'])
def compress_image():
    """Compress image using Huffman coding"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Generate visualization of pixel frequency
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.figure()
        plt.title("Pixel Frequency Distribution")
        plt.plot(hist)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        
        # Save plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        hist_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Perform Huffman compression
        flat_image = image.flatten()
        freq = {}
        for pixel in flat_image:
            freq[pixel] = freq.get(pixel, 0) + 1
        
        huffman_tree = huffman.codebook(freq.items())
        compressed_data = ''.join(huffman_tree[pixel] for pixel in flat_image)
        
        # Create compressed visualization
        compressed_img = cv2.resize(image, (0,0), fx=0.5, fy=0.5)  # Compress size by 50%
        _, compressed_buf = cv2.imencode('.png', compressed_img)
        compressed_base64 = base64.b64encode(compressed_buf).decode('utf-8')
        
        # Calculate compression statistics
        original_size = len(flat_image) * 8  # 8 bits per pixel
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size
        
        result = {
            "status": "success",
            "compression_details": {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "pixel_frequencies": hist_base64,
                "compressed_image": compressed_base64
            }
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logging.error(f"Image compression error: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/bgr', methods=['POST'])
def process_image():
    if 'image' not in request.files or 'background' not in request.files:
        return jsonify({'error': 'No image or background file provided'}), 400

    image_file = request.files['image']
    background_file = request.files['background']

    # Load the main image
    img_bytes = image_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Load the background image
    bgr_img = bgr_image_from_file(background_file)

    # Perform prediction
    output = predict(net, img, bgr_img)
    res_img = post_process(*output, a=True)
    res_img = cv2.cvtColor(res_img, cv2.COLOR_RGBA2BGRA)

    output_path = 'output.png'
    cv2.imwrite(output_path, res_img)

    return send_file(output_path, mimetype='image/png')

@app.errorhandler(Exception)
def handle_error(error):
    response = {
        "error": str(error),
        "message": "An unexpected error occurred"
    }
    return jsonify(response), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)