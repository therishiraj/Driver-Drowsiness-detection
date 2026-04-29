# Deployment Guide

## The Webcam Challenge

Driver drowsiness detection requires **webcam access**, which means:
1. The browser must be on **HTTPS** to access the camera (security requirement)
2. The actual video processing happens on the **client side** (in the user's browser) OR
3. The Flask server streams the already-processed camera feed (requires the server to have a camera — fine for local/demo use)

This guide covers both approaches.

---

## 🏠 Option 1: Local Run (Best for demo & development)

```bash
python app.py
# http://localhost:5000
```

For LAN demo (e.g., showing on a laptop):
```bash
python app.py --host 0.0.0.0
# Other devices: http://YOUR_IP:5000
```

---

## ☁️ Option 2: Render.com (Free tier, HTTPS, Python supported)

**Best for:** Showing a live demo with your own webcam processed server-side.

1. Push your project to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect GitHub → select repo
4. Configure:
   ```
   Runtime:       Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
   ```
5. Add to requirements.txt:
   ```
   gunicorn>=21.0.0
   ```
6. Deploy — you get a free `*.onrender.com` HTTPS URL

> ⚠️ Render's free tier servers have **no camera**. The video feed will show a black/error frame. For a real demo, use the local version OR the client-side JS version.

---

## ⚡ Option 3: Vercel / Netlify (Static + Client-Side JS)

Since Vercel/Netlify host static sites, you need to move drowsiness detection to **client-side JavaScript using TensorFlow.js + face-api.js**.

### Steps:
1. Convert your trained CNN to TensorFlow.js format:
   ```bash
   pip install tensorflowjs
   tensorflowjs_converter --input_format=keras model/eye_cnn.h5 static/tfjs_model/
   ```

2. Use `face-api.js` for landmark detection in browser:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/face-api.js"></script>
   ```

3. Deploy the `static/` folder + `templates/index.html` as a static site.

Full client-side implementation is a separate project — ask for the JS-only version if needed.

---

## 🐳 Option 4: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t drowsiness-detection .
docker run -p 5000:5000 --device /dev/video0 drowsiness-detection
```

The `--device /dev/video0` mounts your webcam into the container.

---

## 🐙 GitHub Setup

```bash
git init
git add .
git commit -m "feat: Real-Time Driver Drowsiness Detection — Intelligent Agent"
git remote add origin https://github.com/YOUR_USERNAME/drowsiness-detection.git
git branch -M main
git push -u origin main
```

### Recommended GitHub Topics:
`python` `opencv` `flask` `deep-learning` `cnn` `computer-vision` `drowsiness-detection` `intelligent-agents` `mtech-project`

---

## 📋 Recommended Approach for Portfolio

| Goal | Recommendation |
|------|---------------|
| GitHub portfolio | Push to GitHub with this README |
| Live demo link | Record a screen capture video, link in README |
| Vercel/Netlify | Deploy the dashboard UI only (static), note camera needs local run |
| Real deployment | Render.com with gunicorn |

---

## 🎬 Recording a Demo

```bash
# On macOS:
# Use QuickTime → File → New Screen Recording → select window

# On Linux:
sudo apt install obs-studio
# or: ffmpeg -video_size 1280x720 -framerate 30 -f x11grab -i :0.0 demo.mp4

# On Windows:
# Win + G → Xbox Game Bar → Record
```
