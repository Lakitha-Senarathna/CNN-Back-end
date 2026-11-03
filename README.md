# Skin Cancer Classification API

A lightweight REST API built with [FastAPI](https://fastapi.tiangolo.com/) that serves a convolutional neural network (CNN) for binary skin-lesion image classification. The API loads the included Keras model, accepts an image upload, and returns a predicted class and confidence score.

> **Medical disclaimer:** This project is for educational and research purposes only. Its output is not a medical diagnosis and must not be used to make healthcare decisions. A qualified medical professional should evaluate any skin lesion of concern.

## Features

- FastAPI-based HTTP API.
- Loads the trained model from `skin_cancer_best_model.h5` when the application starts.
- Accepts an image as a multipart file upload at `POST /predict`.
- Returns one of two labels: `Benign` or `Malignant`, together with the model's confidence percentage.
- Interactive API documentation is available through FastAPI's Swagger UI and ReDoc.

## Repository contents

```text
.
├── main.py                    # FastAPI application and prediction endpoint
├── skin_cancer_best_model.h5  # Trained Keras model
└── .gitignore
```

The model file is approximately 45 MB and must be present at the path expected by the application.

## Requirements

- Python 3.9 or newer (use a Python version supported by the TensorFlow release installed for your platform).
- pip.
- The dependencies used by the application:
  - FastAPI
  - Uvicorn
  - TensorFlow
  - Pillow
  - NumPy
  - `python-multipart` (required by FastAPI to parse uploaded files)

## Setup and run

1. Clone the repository and move into its directory:

   ```bash
   git clone https://github.com/Lakitha-Senarathna/CNN-Back-end.git
   cd CNN-Back-end
   ```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install fastapi "uvicorn[standard]" tensorflow pillow numpy python-multipart
   ```

   TensorFlow installation can vary by operating system, processor, and Python version. If installation fails, consult the [official TensorFlow installation instructions](https://www.tensorflow.org/install) and install a compatible release.

4. Start the development server from the repository root (the model is loaded using a path relative to the current working directory):

   ```bash
   python -m uvicorn main:app --reload
   ```

   The API will be available at `http://127.0.0.1:8000` by default. To listen on another interface or port, for example:

   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   Only expose the server to a trusted network unless you have added appropriate security and operational controls.

## API reference

### `GET /`

A basic readiness response. The application loads the model during startup/import; if loading fails, the server may not start.

Example:

```bash
curl http://127.0.0.1:8000/
```

Example response:

```json
{
  "message": "Hello World. Model is loaded and ready."
}
```

### `POST /predict`

Submit an image using `multipart/form-data` with the field name `file`.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/skin-lesion-image.jpg"
```

A successful response has this shape:

```json
{
  "filename": "skin-lesion-image.jpg",
  "diagnosis": "Benign",
  "confidence": 97.42
}
```

- `filename`: uploaded filename supplied by the client.
- `diagnosis`: predicted class (`Benign` or `Malignant`).
- `confidence`: maximum model output, expressed as a percentage and rounded to two decimal places. This is a model score, not a clinically validated probability.

The application currently does not define custom error responses for invalid images, unsupported formats, oversized uploads, or inference failures. FastAPI's interactive documentation is available at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Model and preprocessing

The model is loaded from `skin_cancer_best_model.h5` and the application declares the class order as `Benign`, then `Malignant`. The preprocessing function resizes images to 224 × 224 pixels and contains code intended to normalize pixel values to the 0–1 range.

> **Current implementation caveat:** In `main.py`, the preprocessing function calculates a NumPy array but returns an expanded PIL image object instead of the processed array. The calculated normalization is therefore not passed to the model. Prediction requests may fail or produce incorrect results until preprocessing is corrected and verified against the model's training-time input requirements. Confirm the expected color channels, shape, dtype, and normalization with the model before relying on predictions.

## Development notes

- Run Uvicorn from the repository root so the relative model path resolves.
- The model is loaded once when the application module is imported; model load time and memory use depend on the environment.
- Before deployment, consider adding input validation, upload-size limits, structured error handling, logging, automated tests, and an explicit health/readiness check.
- Do not upload sensitive or personally identifiable medical images to an untrusted deployment.

## License

No license file is currently included in this repository. Unless the repository owner adds a license, all rights remain reserved and reuse is not automatically permitted.
