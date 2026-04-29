"""
Real-Time Driver Drowsiness Detection System
Intelligent Agent Architecture: Perception → Reason → Action → Feedback
"""

from flask import Flask, render_template, Response, jsonify, request
from agent.drowsiness_agent import DrowsinessAgent
import cv2
import threading
import base64
import numpy as np

app = Flask(__name__)

# Global agent instance
agent = DrowsinessAgent()
agent_lock = threading.Lock()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream processed video frames with drowsiness overlay."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def generate_frames():
    """Generator that yields MJPEG frames from the agent's camera."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with agent_lock:
                processed_frame, state = agent.process_frame(frame)

            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
    finally:
        cap.release()


@app.route('/agent_state')
def agent_state():
    """Return current agent perception/reasoning state as JSON."""
    with agent_lock:
        state = agent.get_state()
    return jsonify(state)


@app.route('/reset_agent', methods=['POST'])
def reset_agent():
    """Reset agent internal state (clears alert history)."""
    with agent_lock:
        agent.reset()
    return jsonify({'status': 'reset', 'message': 'Agent state cleared.'})


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'agent': 'DrowsinessAgent v1.0'})


if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
