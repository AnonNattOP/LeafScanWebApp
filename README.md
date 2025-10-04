# Plant Disease Detection Web App

This project packages a trained EfficientNet-B0 model into a simple Flask web application. Upload a plant leaf photo in the browser and the app responds with the most likely disease label plus class probabilities. The repository already contains the trained weights (`efficientnet_b0_plant_disease.pth`) and a scaffolded front end so you can get started quickly.

## Prerequisites
- Python 3.10 (or newer 3.x release) installed on your machine
- A terminal with `python` and `pip` on the PATH (PowerShell, Command Prompt, or similar)
- The model checkpoint file: `efficientnet_b0_plant_disease.pth` (already in this folder)

> Tip: If you are on Windows, run the commands below from *PowerShell* in the project directory `c:\Users\Natta\Documents\univer\year3\DeepLerning\project`.

## 1. Create a virtual environment (recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
If activation succeeds, your prompt shows `(.venv)` in front of the path. To exit later, run `deactivate`.

## 2. Install dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
This installs Flask, Torch, Torchvision, and Pillow. The first Torch install can take a few minutes because it downloads a large wheel.

## 3. Configure class labels
The app needs human-readable labels for each output neuron. Edit `class_names.json` and list the class names **in the same order** that the model was trained. Example:
```json
[
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust"
]
```
- If you leave the list empty (`[]`), the UI falls back to generic names such as `Class 0`, `Class 1`, etc.
- You can find the correct order by checking the folder names used when training (`datasets.ImageFolder` sorts alphabetically).

## 4. Launch the web app
With the virtual environment activated, run:
```powershell
python app.py
```
Flask starts in debug mode and prints a URL similar to `http://127.0.0.1:5000`. Open that address in your browser, choose an image, and submit.

### Optional: Use cURL for quick checks
```powershell
curl -F "image=@path\to\sample.jpg" http://127.0.0.1:5000/predict
```
The response is a JSON object with the predicted label and probability list.

## 5. Project structure
```
project/
├── app.py                # Flask entrypoint
├── model.py              # EfficientNet loader + preprocessing
├── efficientnet_b0_plant_disease.pth
├── class_names.json      # Editable label list
├── templates/index.html  # Upload form + UI
└── static/styles.css     # Styling for the page
```

## Troubleshooting
- **Torch not installed / missing CUDA DLLs**: Re-run `pip install -r requirements.txt`. If you do not need GPU support, CPU wheels are installed automatically.
- **App cannot find class labels**: Ensure `class_names.json` contains exactly the same number of entries as the model outputs.
- **Checkpoint mismatch errors**: Confirm that `efficientnet_b0_plant_disease.pth` matches the EfficientNet-B0 architecture. The included `model.py` expects the classifier to be the fine-tuned head from `sai_kiew.ipynb`.
- **Large image uploads**: For production use you may want to reduce file size or add validation checks. The current demo reads the entire file into memory.

Happy experimenting!
# LeafScanWebApp
# LeafScanWebApp
