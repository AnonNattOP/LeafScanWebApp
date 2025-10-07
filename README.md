# Plant Disease Detection Web App

This repository wraps a fine-tuned EfficientNet-B0 model in a lightweight Flask interface. Point it at a photo of a plant leaf and it returns the disease label along with the probability scores. The trained weights (`efficientnet_b0_plant_disease.pth`) and the basic front end are already included, so you can focus on running or customizing the app.

## Quick Start
1. **Install Python 3.10 or newer.** Verify it works with `python --version`.
2. **Download or clone this folder** to a convenient location.
3. **Set up a virtual environment** (recommended) and install the dependencies from `requirements.txt`.
4. **Check `class_names.json`.** It already contains the 15 labels the model was trained on; edit it only if you swap in a different checkpoint.
5. **Run `python app.py`** and open the printed URL in your browser. The page will state whether the plant looks healthy or list the most likely disease.

## Step-by-step setup

### 1. Create and activate a virtual environment
Pick the commands that match your operating system.

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

When activation works, your prompt shows `(.venv)` before the current path. Deactivate with `deactivate` when you are done.

### 2. Install the Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
The first PyTorch install may take a few minutes because it downloads a large wheel. Add `--user` if you prefer per-user installs, and use `pip3` when your system distinguishes Python 3 from Python 2. All required packages (Flask, Torch, Torchvision, Pillow) are listed in `requirements.txt`.

### 3. Review the class labels
`class_names.json` stores the human-readable labels that appear in the UI.
- The file is already populated with the tomato, potato, and pepper classes that match the bundled checkpoint.
- If you bring your own model, replace the entries so they match the new output order exactly.
- If the list is empty or the counts do not match the model outputs, the app falls back to generic names like `Class 0`.

> Tip: If you trained with `torchvision.datasets.ImageFolder`, the classes are sorted alphabetically by folder name, so you can copy the order straight from your training dataset.

### 4. Launch the web application
```bash
python app.py
```
Flask starts in debug mode and prints a URL such as `http://127.0.0.1:5000`. Open it in a browser, choose a leaf image, and the UI will report “The plant appears healthy...” or “This plant is most likely affected by...” along with confidence scores for all classes.

#### Optional: test the API from the command line
```bash
curl -F "image=@path/to/leaf.jpg" http://127.0.0.1:5000/predict
```
The server responds with a JSON payload containing the predicted label and the probability list.

## Dataset snapshot (for retraining only)
The repository includes `dataset_split/` as a reference to the tomato, potato, and pepper classes used during training. You do **not** need these files to run the web app, but they are handy if you want to fine-tune the model again or regenerate label files.

```
dataset_split/
|-- train/
|   |-- Pepper__bell___Bacterial_spot/
|   |-- Pepper__bell___healthy/
|   |-- Potato___Early_blight/
|   |-- Potato___healthy/
|   |-- Potato___Late_blight/
|   |-- Tomato_Bacterial_spot/
|   |-- Tomato_Early_blight/
|   |-- Tomato_healthy/
|   |-- Tomato_Late_blight/
|   |-- Tomato_Leaf_Mold/
|   |-- Tomato_Septoria_leaf_spot/
|   |-- Tomato_Spider_mites_Two_spotted_spider_mite/
|   |-- Tomato__Target_Spot/
|   |-- Tomato__Tomato_mosaic_virus/
|   `-- Tomato__Tomato_YellowLeaf__Curl_Virus/
|-- val/
|   `-- ... (same class folders as train)
`-- test/
    `-- ... (same class folders as train)
```

### Retrain or update the model
Use the `sai_kiew.ipynb` notebook to fine-tune EfficientNet-B0 on the bundled dataset:
1. Open `sai_kiew.ipynb` in Jupyter Notebook or VS Code.
2. Run the notebook; it loads data from `dataset_split/`, trains the model, and exports a new checkpoint (for example, `efficientnet_b0_plant_disease.pth`).
3. Copy the new `.pth` file into the project root (overwrite the old one if desired).
4. Regenerate `class_names.json` if the class order changed.

### Regenerate `class_names.json` when you retrain
Run this one-liner from the project root to dump the alphabetical class list into `class_names.json` based on your training split:

```bash
python -c "from pathlib import Path; classes = sorted(p.name for p in Path('dataset_split/train').iterdir() if p.is_dir()); import json, sys; json.dump(classes, sys.stdout, indent=4)"
```

Update the command to match your dataset path if you store data elsewhere.

## Project structure
```
project/
|-- app.py                # Flask entry point and HTTP endpoints
|-- model.py              # EfficientNet loader and inference helpers
|-- efficientnet_b0_plant_disease.pth
|-- class_names.json      # Human-readable labels shown in the UI
|-- requirements.txt      # Python dependencies used by the app
|-- dataset_split/        # Optional reference dataset for retraining
|-- templates/
|   `-- index.html        # Upload form, fetch logic, result rendering
|-- static/
|   `-- styles.css        # Basic styling for the web UI
`-- sai_kiew.ipynb        # Notebook used to retrain EfficientNet-B0
```

**File reference**
- `app.py`: Boots the Flask app, exposes `/` for the UI and `/predict` for JSON inference.
- `model.py`: Wraps the EfficientNet-B0 weights, handles preprocessing, inference, and label lookups.
- `efficientnet_b0_plant_disease.pth`: Fine-tuned checkpoint exported from training (e.g., `sai_kiew.ipynb`).
- `class_names.json`: Ready-to-use labels aligned with the model's output order.
- `dataset_split/`: Reference dataset split so you can retrain or regenerate labels (not required for inference).
- `templates/index.html`: Front-end page that collects the image and posts it to `/predict`.
- `static/styles.css`: Styling for the single-page UI.
- `requirements.txt`: Pin list for Flask, Torch, Torchvision, Pillow, etc.
- `sai_kiew.ipynb`: Jupyter notebook that trains EfficientNet-B0 on `dataset_split/`.

## Model analysis
- **Backbone:** EfficientNet-B0 fine-tuned on 15 tomato, potato, and pepper classes. model.py rebuilds the classifier head to match the checkpoint and keeps the network in eval mode on CPU or CUDA depending on availability.
- **Preprocessing:** Images are converted to RGB, resized to 224x224 pixels, turned into tensors, and normalized with ImageNet statistics (means 0.485/0.456/0.406, stds 0.229/0.224/0.225). Training-time augmentation lives in sai_kiew.ipynb; inference stays deterministic.
- **Outputs:** The /predict endpoint returns the top label, its confidence, and the full softmax probability list. Class display strings still originate from class_names.json, so keep that file aligned with the checkpoint outputs.
- **Performance notes:** EfficientNet-B0 balances accuracy and speed for CPU inference. Swap in a heavier EfficientNet variant for better accuracy or prune or distill the model if latency becomes an issue.
- **Limitations:** Accuracy degrades with blurry photos, mixed leaves, or classes not present in the training set. The model expects leaf-centric shots similar to those in dataset_split.

## Web structure analysis
- **Server side:** app.py initialises a single PlantDiseaseModel, serves the root route with index.html, and exposes /predict for JSON inference. It also loads class_names_display.json so the UI can show curated, user-friendly names.
- **Template:** templates/index.html renders the upload flow, preview, prediction block, class list, and an information panel with causes, symptoms, and treatments. Vanilla JavaScript handles file reading, fetch requests, and DOM updates.
- **Static assets:** static/styles.css provides the glassmorphism styling, grid layouts for good and bad photo examples, and responsive tweaks. Helpful images live under static/img/.
- **Data flow:** The browser submits a FormData payload to /predict, receives JSON with probabilities, and updates the UI. When display labels exist, the class list uses them while inference still relies on class_names.json.
- **Extensibility:** Add features by extending app.py, adjusting the template, and editing the stylesheet. For production, front the app with Gunicorn or another WSGI server and add logging, authentication, and upload validation.

## Troubleshooting & FAQs
- **PyTorch fails to install or complains about CUDA:** Try rerunning `pip install -r requirements.txt`. CPU wheels install by default; GPU support is optional.
- **Runtime error about label length:** Confirm `class_names.json` has the same number of entries as the model outputs.
- **State dict mismatch when loading the checkpoint:** Ensure you are using the provided EfficientNet-B0 checkpoint or retrain and update the classifier definition in `model.py`.
- **Large image uploads cause slow responses:** Compress or resize images before uploading, or add validation logic in `app.py` for production use.

### PowerShell activation blocked by a revoked certificate
Older Python releases signed the virtual-environment activation script with a certificate that has since been revoked. PowerShell therefore stops on `.\.venv\Scripts\Activate.ps1` with `A certificate was explicitly revoked by its issuer`.

1. Prefer installing a newer Python build (3.10.11+, 3.11.3+, 3.12.x), delete `.venv`, and recreate it with `python -m venv .venv`. The fresh `Activate.ps1` carries a valid signature.
2. Short-term, run PowerShell with a relaxed execution policy (`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) before activating, or use `cmd` and run `.venv\Scripts\activate.bat`.
3. As a last resort, remove the signature block at the bottom of `.venv\Scripts\Activate.ps1` (between `# SIG # Begin signature block` and `# SIG # End signature block`) so it runs under `RemoteSigned`.

## Next steps
- Swap in a different checkpoint and update `class_names.json` to match your new classes.
- Customize `templates/index.html` and `static/styles.css` to improve the UI.
- For deployment, run behind a production-ready server (Gunicorn, Uvicorn + ASGI adapter) and add logging, HTTPS, and request limits.

Happy experimenting!
