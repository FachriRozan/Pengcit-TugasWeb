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

def scale_image(image, method):
    """Scale image using different interpolation methods"""
    height, width = image.shape[:2]
    
    # More aggressive downscaling for better visibility of differences
    small_height = int(height * 0.15)  # Downscale to 15%
    small_width = int(width * 0.15)
    
    # Larger upscaling for more noticeable differences
    target_height = height * 3  # Scale up to 3x
    target_width = width * 3

    if method == 'nearest':
        # Nearest Neighbor - most basic interpolation
        small_image = cv2.resize(image, (small_width, small_height), 
                               interpolation=cv2.INTER_NEAREST)
        result = cv2.resize(small_image, (target_width, target_height), 
                          interpolation=cv2.INTER_NEAREST)
        
    elif method == 'linear':
        # Linear interpolation (1D) - only horizontal interpolation
        # First resize only width
        temp1 = cv2.resize(image, (small_width, height),
                         interpolation=cv2.INTER_LINEAR)
        # Then resize height using nearest neighbor
        small_image = cv2.resize(temp1, (small_width, small_height),
                               interpolation=cv2.INTER_NEAREST)
        
        # Upscale using the same approach
        temp2 = cv2.resize(small_image, (target_width, small_height),
                          interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(temp2, (target_width, target_height),
                          interpolation=cv2.INTER_NEAREST)
        
    elif method == 'bilinear':
        # Bilinear interpolation (2D) - both directions
        small_image = cv2.resize(image, (small_width, small_height), 
                               interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small_image, (target_width, target_height),
                          interpolation=cv2.INTER_LINEAR)
        
    elif method == 'cubic':
        # Bicubic interpolation - smoother results
        small_image = cv2.resize(image, (small_width, small_height), 
                               interpolation=cv2.INTER_CUBIC)
        result = cv2.resize(small_image, (target_width, target_height),
                          interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Unsupported scaling method")

    return result

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
    
    # Read image
    img = Image.open(file.stream)
    img_np = np.array(img)
    
    # Convert RGB to BGR for OpenCV
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Scale image
    scaled = scale_image(img_np, method)
    
    # Convert back to RGB
    if len(scaled.shape) == 3:
        scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image and save to buffer
    result_image = Image.fromarray(scaled)
    img_io = BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)