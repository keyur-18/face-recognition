# 🧠 Face Recognition App

A **real-time face recognition web application** built with [Streamlit](https://streamlit.io/) and powered by [DeepFace](https://github.com/serengil/deepface) using the **ArcFace** model. The app supports enrolling new users and recognizing faces through a camera input — all within a clean, multi-page interface.

---

## ✨ Features

- 📷 **Face Enrollment** — Capture 3 face images (front, left, right angles) and save them with a username
- 🔍 **Face Recognition** — Capture a live image and identify the person from the enrolled database
- 💾 **Embedding Storage** — Face embeddings are stored persistently in a JSON database (`face_embeddings.json`)
- 🐳 **Docker Support** — Fully containerized for easy deployment
- ✅ **CI/CD Pipeline** — Automated testing via GitHub Actions

---

## 🛠️ Tech Stack

| Component        | Technology                        |
|-----------------|-----------------------------------|
| Frontend UI      | Streamlit                         |
| Face Recognition | DeepFace (ArcFace model)          |
| Similarity Metric| Cosine Distance (scipy)           |
| Image Processing | OpenCV, Pillow, NumPy             |
| Backend API      | FastAPI + Uvicorn                 |
| Containerization | Docker (multi-stage build)        |
| Testing          | Pytest + pytest-mock              |
| CI/CD            | GitHub Actions                    |

---

## 📁 Project Structure

```
face recognition/
├── app.py                    # Main Streamlit entry point (multi-page navigation)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production Docker image (multi-stage build)
├── Dockerfile.base           # Base Docker image with system dependencies
│
├── pages/
│   ├── face_enroll.py        # Face enrollment page (capture & save 3 images)
│   └── face_recog.py         # Face recognition page (identify from camera)
│
├── utils/
│   ├── embed.py              # Core embedding logic (create, save, recognize)
│   └── enroll_utils.py       # Enrollment helpers (validate, save images)
│
├── model/                    # ⚠️ Not tracked — download separately (see below)
│   ├── arc.onnx              # ArcFace ONNX model weights
│   └── yolov8n-face.pt       # YOLOv8 face detection model
│
├── tests/
│   ├── test_embed.py         # Unit tests for identity/embedding logic
│   ├── test_enroll_utils.py  # Unit tests for enrollment utilities
│   └── test_integration.py   # Integration tests
│
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions CI pipeline
```

> **Note:** The `images/`, `model/`, and `face_embeddings.json` directories are excluded from version control via `.gitignore`. See below for model download instructions.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- A working webcam (for capturing face images)

### 1. Clone the Repository

```bash
git clone https://github.com/keyur-18/face-recognition.git
cd face-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models

The `model/` folder is not included in this repository. Download the required model files and place them inside the `model/` directory:

📥 **[Download Models](<!-- ADD YOUR LINK HERE -->)**

```
model/
├── arc.onnx           # ArcFace face recognition model
└── yolov8n-face.pt    # YOLOv8 face detection model
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🐳 Running with Docker

### Build and Run

```bash
docker build -t face-recognition-app .
docker run -p 8501:8501 face-recognition-app
```

Then navigate to `http://localhost:8501` in your browser.

### Build the Base Image (optional)

The base image pre-installs all heavy system dependencies for faster rebuilds:

```bash
docker build -f Dockerfile.base -t keyur18/face-recognition-base .
```

---

## 📖 How It Works

### Face Enrollment (`/face enroll`)

1. Enter your **username**
2. Capture **3 face images** — Front, Left angle, Right angle
3. Click **Save Images**
4. The app extracts **512-dimensional ArcFace embeddings** for each image and saves them to `face_embeddings.json`

### Face Recognition (`/face recognition`)

1. Capture a **live image** from your camera
2. Click **Generate**
3. The app computes an embedding for the captured face
4. It compares against all stored embeddings using **cosine distance**
5. The closest match (below the similarity threshold) is returned as the recognized identity

### Identity Matching Logic

- Embeddings are **L2-normalized** before comparison
- Cosine distance is used as the similarity metric
- A **threshold** (`0.5`) filters out unrecognized faces
- A **margin** (`0.05`) ensures the best match is clearly better than the second-best to reduce false positives
- Returns `"unknown"` if no confident match is found

---

## 🧪 Running Tests

```bash
pytest -v
```

Tests cover:
- ✅ Correct face identity matching
- ✅ Unknown face detection
- ✅ Zero-vector edge case handling
- ✅ Nearest-person identification with noise

---

## ⚙️ CI/CD

The project uses **GitHub Actions** for continuous integration:

- **Trigger**: Push to `testing` branch or pull request to `main`
- **Environment**: Runs on `keyur18/face-recognition-base` Docker container
- **Step**: Executes `pytest -v` to validate all tests pass

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full pipeline config.

---

## 📦 Dependencies

```
streamlit          # Web UI framework
deepface           # Face recognition (ArcFace model)
numpy              # Numerical operations
scipy              # Cosine distance calculation
pillow             # Image handling
opencv-python      # Image processing
tf-keras           # Keras backend for DeepFace
fastapi            # REST API backend
uvicorn            # ASGI server
pytest             # Testing framework
pytest-mock        # Mock utilities for tests
python-multipart   # Multipart form data support
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request to `main`

---

## 📄 License

This project is open-source. Feel free to use and modify it for your own purposes.

---

> Built with ❤️ using Streamlit & DeepFace
