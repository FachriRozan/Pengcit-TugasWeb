from flask import Flask, request, jsonify, send_file
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

# Inisialisasi Flask app
matplotlib.use('Agg')
app = Flask(__name__)
CORS(app)  # Izinkan semua asal untuk semua endpoint
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load model DeepLabV3 yang sudah dilatih
SegModel = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

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

# Image Deblurring 
def deblur_image(image, blur_type):
    if blur_type == 'motion':
        kernel = np.zeros((15, 15))
        kernel[int((15 - 1) / 2), :] = np.ones(15)
        kernel = kernel / 15
    elif blur_type == 'linear_motion':
        kernel = np.zeros((15, 15))
        kernel[:, int((15 - 1) / 2)] = np.ones(15)
        kernel = kernel / 15
    elif blur_type == 'gaussian':
        kernel = cv2.getGaussianKernel(15, 0)
        kernel = kernel * kernel.T
    elif blur_type == 'defocus':
        kernel = np.zeros((15, 15))
        cv2.circle(kernel, (int(15 / 2), int(15 / 2)), 7, 1, -1)
        kernel = kernel / np.sum(kernel)
    elif blur_type == 'atmospheric':
        kernel = np.ones((15, 15)) / (15 * 15)
    else:
        raise ValueError("Unsupported blur type")

    deblurred = cv2.filter2D(image, -1, kernel)
    return deblurred

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
@app.route('/api/edge-detection', methods=['POST'])
def edge_detection():
    file = request.files['image']
    edge_method = request.form.get('edge_method', 'canny')  # Get edge method from form data, default to 'canny'

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply edge detection
    edges = apply_edge_detection(img_cv, edge_method)

    # Convert result back to PIL for sending as response
    img_pil = Image.fromarray(edges)
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

# API for RGB histogram
@app.route('/histogram/original', methods=['POST'])
def original_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Calculate RGB histogram
    hist_buf = calculate_rgb_histogram(img_np)

    return send_file(hist_buf, mimetype='image/png')

# API for equalized histogram
@app.route('/histogram/equalized', methods=['POST'])
def equalized_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Perform histogram equalization
    equalized_image = equalize_image(img_np, mode='RGB')
    hist_buf = calculate_rgb_histogram(equalized_image)

    return send_file(hist_buf, mimetype='image/png')

# API for grayscale image and its histogram
@app.route('/histogram/grayscale', methods=['POST'])
def grayscale_histogram():
    file = request.files['image']
    img = Image.open(file.stream).convert('L')  # Convert to grayscale
    img_np = np.array(img)

    # Calculate grayscale histogram
    hist_buf = calculate_grayscale_histogram(img_np)

    return send_file(hist_buf, mimetype='image/png')
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

# API for Image Deblurring
@app.route('/api/deblur', methods=['POST'])
def deblur():
    file = request.files['image']
    blur_type = request.form.get('blur_type', 'motion')  # Get blur type from form data, default to 'motion'

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply deblurring
    deblurred_image = deblur_image(img_cv, blur_type)

    # Convert result back to PIL for sending as response
    img_pil = Image.fromarray(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
    img_io = BytesIO()
    img_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)