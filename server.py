from flask import Flask, request, jsonify
import whisper
import torch
import os
import tempfile
import logging
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model once at startup
try:
    logger.info("Loading Whisper model...")
    MODEL = whisper.load_model("turbo", device=device)
    logger.info(f"Model loaded successfully on {MODEL.device}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}", exc_info=True)
    raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(MODEL.device),
        'numpy_version': np.__version__,
        'torch_version': torch.__version__
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if not audio_file.filename:
            return jsonify({'success': False, 'error': 'Empty file provided'}), 400

        logger.info(f"Processing audio file: {audio_file.filename}")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            logger.info(f"Saved audio to temporary file: {temp_file.name}")
            
            try:
                # Transcribe with optimized settings
                result = MODEL.transcribe(
                    temp_file.name,
                    fp16=False,
                    language='en',
                    without_timestamps=True,
                    initial_prompt="This is a voice message."
                )
                
                logger.info("Transcription completed successfully")
                
                return jsonify({
                    'success': True,
                    'text': result['text'],
                    'segments': result.get('segments', [])
                })
                
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500
            
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)