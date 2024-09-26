from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torchvision import models
from PIL import Image
import numpy as np
import io
import torchvision.transforms as T

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app, resources={r"/segment": {"origins": "*"}})  # Izinkan semua asal untuk endpoint ini

# Load model DeepLabV3 yang sudah dilatih
SegModel = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

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
    trf = T.Compose([T.Resize(256),
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

# Endpoint untuk menerima dan memproses gambar
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

if __name__ == '__main__':
    app.run(debug=True)
