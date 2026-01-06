from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import logging
from werkzeug.utils import secure_filename

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = Flask(__name__)

# Security & File Upload Settings
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
IMG_SIZE = 128
FRAME_SKIP = 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD MODEL (CACHED)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    model = tf.keras.models.load_model(
        "model/deepfake_model.h5",
        compile=False
    )
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    model = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITY FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def preprocess_image(img):
    """Preprocess image for model prediction"""
    try:
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        return np.expand_dims(img, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def get_prediction_result(prediction_value):
    """Convert prediction value to label and confidence"""
    is_real = prediction_value > 0.5
    confidence = float(prediction_value if is_real else 1 - prediction_value) * 100
    label = "REAL" if is_real else "FAKE"
    return label, round(confidence, 2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        "status": "ok",
        "model": model_status
    }), 200

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMAGE PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/predict-image", methods=["POST"])
def predict_image():
    """
    Image detection endpoint
    Accepts: POST request with image file
    Returns: JSON with prediction, confidence, and analysis details
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["file"]
    
    # Validate filename
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({
            "error": f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        }), 400
    
    try:
        # Load and preprocess image
        img = Image.open(file).convert("RGB")
        processed_img = preprocess_image(img)
        
        if processed_img is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Make prediction
        prediction = model.predict(processed_img, verbose=0)[0][0]
        label, confidence = get_prediction_result(prediction)
        
        logger.info(f"Image prediction: {label} ({confidence}%)")
        
        return jsonify({
            "label": label,
            "confidence": confidence,
            "filename": secure_filename(file.filename)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in image prediction: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIDEO PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/predict-video", methods=["POST"])
def predict_video():
    """
    Video detection endpoint
    Analyzes video frames and detects deepfakes
    Accepts: POST request with video file
    Returns: JSON with overall prediction, frame analysis, and confidence
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    
    file = request.files["file"]
    
    # Validate filename
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Validate file extension
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({
            "error": f"Invalid video format. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        }), 400
    
    video_path = None
    cap = None
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        logger.info(f"Processing video: {filename}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 400
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return jsonify({"error": "Video has no frames"}), 400
        
        # Analyze frames
        fake_frames = 0
        real_frames = 0
        frame_count = 0
        frames_analyzed = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for efficiency
            if frame_count % FRAME_SKIP != 0:
                continue
            
            try:
                # Preprocess frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame.astype("float32") / 255.0
                frame = np.expand_dims(frame, axis=0)
                
                # Predict
                pred = model.predict(frame, verbose=0)[0][0]
                
                if pred < 0.5:
                    fake_frames += 1
                else:
                    real_frames += 1
                
                frames_analyzed += 1
            
            except Exception as e:
                logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                continue
        
        cap.release()
        
        # Calculate results
        if frames_analyzed == 0:
            return jsonify({"error": "No frames could be analyzed"}), 400
        
        # Determine overall label
        label = "FAKE" if fake_frames > real_frames else "REAL"
        confidence = round(max(fake_frames, real_frames) / frames_analyzed * 100, 2)
        
        logger.info(f"Video analysis: {label} ({confidence}%) - {frames_analyzed} frames analyzed")
        
        return jsonify({
            "label": label,
            "confidence": confidence,
            "frames_analyzed": frames_analyzed,
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "filename": filename
        }), 200
    
    except Exception as e:
        logger.error(f"Error in video prediction: {str(e)}")
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        
        # Delete uploaded file after processing
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Deleted temporary file: {video_path}")
            except Exception as e:
                logger.warning(f"Could not delete file: {str(e)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ERROR HANDLERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({"error": "File too large. Maximum size is 500MB"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 error"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 error"""
    return jsonify({"error": "Internal server error"}), 500

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN SERVER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    logger.info("🚀 Starting DeepShield Pro server...")
    logger.info(f"📍 Server running on 0.0.0.0:{port}")
    logger.info("🔧 Debug mode: OFF (production)")

    app.run(
        debug=False,          
        host="0.0.0.0",       
        port=port,
        threaded=True
    )
