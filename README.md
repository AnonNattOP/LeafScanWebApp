# Plant Disease Detection Web App

## Project Overview
This repository wraps a fine-tuned EfficientNet-B0 model in a lightweight Flask interface. Point it at a photo of a plant leaf and it returns the disease label along with the probability scores. The trained weights (`efficientnet_b0_plant_disease.pth`) and the basic front end are already included, so you can focus on running or customizing the app.

- Inference-ready EfficientNet-B0 weights are bundled so you can run predictions immediately.
- A minimal Flask backend serves both the browser front end and a JSON `/predict` endpoint.
- The responsive UI in `templates/index.html` visualises confidence scores and shares care tips.

## Requirements
- Python 3.10 or newer with `pip`
- Git (optional, for cloning)
- A virtual environment tool such as `venv` (recommended)
- `curl` (optional, for command-line endpoint testing)

## Installation
### 1. Obtain the project files
- Clone the repository or download it as an archive and extract it.
- Ensure the model checkpoint (`efficientnet_b0_plant_disease.pth`) and label map (`class_names.json`) stay in the project root.

```powershell
# Windows (PowerShell)
git clone <repository-url>
cd project
```

```bash
# macOS / Linux
git clone <repository-url>
cd project
```

If you downloaded a ZIP, unzip it and open a shell in the extracted folder instead of running `git clone`.

### 2. Create and activate a virtual environment
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

### 3. Install the Python packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
The first PyTorch install may take a few minutes because it downloads a large wheel. Add `--user` if you prefer per-user installs, and use `pip3` when your system distinguishes Python 3 from Python 2. All required packages (Flask, Torch, Torchvision, Pillow) are listed in `requirements.txt`.

### 4. Validate the bundled assets
`class_names.json` stores the human-readable labels that appear in the UI.
- The file is already populated with the tomato, potato, and pepper classes that match the bundled checkpoint.
- If you bring your own model, replace the entries so they match the new output order exactly.
- If the list is empty or the counts do not match the model outputs, the app falls back to generic names like `Class 0`.

> Tip: If you trained with `torchvision.datasets.ImageFolder`, the classes are sorted alphabetically by folder name, so you can copy the order straight from your training dataset.

### 5. Capture the environment (recommended)
For strict reproducibility, freeze the environment after installation so you can recreate it later or on another machine.

```bash
pip freeze > requirements.lock
```
Commit the generated `requirements.lock` file or store it alongside experiment notes.

## Running the web application
### Launch the development server
```bash
python app.py
```
Flask starts in debug mode and prints a URL such as `http://127.0.0.1:5000`. Open it in a browser, choose a leaf image, and the UI will report "The plant appears healthy..." or "This plant is most likely affected by..." along with confidence scores for all classes.

### Confirm the REST endpoint
```bash
curl -F "image=@path/to/leaf.jpg" http://127.0.0.1:5000/predict
```
The server responds with a JSON payload containing the predicted label and the probability list. Replace `path/to/leaf.jpg` with an actual image of a leaf.

### Shut down
Press `Ctrl+C` in the terminal to stop the Flask development server.

## Reproducibility checklist
- Use the same Python version (3.10+) and install dependencies via `pip install -r requirements.txt`.
- Keep `efficientnet_b0_plant_disease.pth` and `class_names.json` in sync; mismatches cause incorrect predictions.
- Document the `pip freeze` output (see step 5) when you publish results or hand off the project.
- The dataset reference structure in `dataset_split/` shows the exact class names and split used during training.
- Run `python -m pip check` to confirm there are no dependency conflicts before sharing the environment.

## Dataset snapshot (for retraining only)
The repository includes `dataset_split/` as a reference to the tomato, potato, and pepper classes used during training. You do **not** need these files to run the web app, but they are handy if you want to fine-tune the model again or regenerate label files.

### Retrain or update the model
Use the `plant_disease_training.ipynb` notebook to fine-tune EfficientNet-B0 on the bundled dataset:
1. Open `plant_disease_training.ipynb` in Jupyter Notebook or VS Code.
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
|-- app.py                      # Flask entry point and HTTP endpoints
|-- model.py                    # EfficientNet loader and inference helpers
|-- class_names.json            # Label IDs aligned with the checkpoint outputs
|-- class_names_display.json    # Optional display names shown in the UI
|-- efficientnet_b0_plant_disease.pth
|-- requirements.txt            # Runtime dependencies for the web app
|-- Dockerfile                  # Container recipe for reproducible deployments
|-- docker-compose.sai.yml      # Example compose file used during local experiments
|-- plant_disease_training.ipynb
|-- templates/
|   `-- index.html              # Upload form, fetch logic, and UI rendering
|-- static/
|   |-- styles.css              # Glassmorphism styling and responsive rules
|   `-- img/
|       |-- should/             # Example images that yield good predictions
|       `-- shouldnot/          # Example images to avoid
|-- dataset_split/              # Reference dataset (train/val/test folders)
`-- README.md
```
The repository also contains a `.venv/` folder and `__pycache__/` directory when you run the code locally; these artefacts are not required for distribution.

**File reference**
- `app.py`: Boots the Flask app, exposes `/` for the UI and `/predict` for JSON inference, and wires in the model loader.
- `model.py`: Wraps EfficientNet-B0, handles preprocessing, inference, and label lookups for both display and prediction labels.
- `class_names.json`: Canonical class order exported from training; must match the checkpoint tensor order.
- `class_names_display.json`: Optional mapping that supplies friendly names for the front end while predictions still use `class_names.json`.
- `efficientnet_b0_plant_disease.pth`: Fine-tuned EfficientNet-B0 weights ready for inference.
- `templates/index.html`: Front-end page that collects the image, makes fetch calls, and renders results.
- `static/styles.css`: Styling for the single-page UI; references example images under `static/img/`.
- `dataset_split/`: Reference dataset structure (train/val/test) for retraining or regenerating label files; not needed for inference.
- `plant_disease_training.ipynb`: Jupyter notebook that fine-tunes EfficientNet-B0 and exports updated checkpoints.
- `Dockerfile` / `docker-compose.sai.yml`: Containerisation assets to rebuild the environment consistently across machines.

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

### Demo


https://github.com/user-attachments/assets/adfaa80f-0bc2-4244-833b-de720a8da9c8

https://github.com/user-attachments/assets/e2153f39-115e-4401-9f6d-02fad914ae6a

https://github.com/user-attachments/assets/4f46a6f8-5664-4fd7-920e-3459c73e16fc

https://github.com/user-attachments/assets/1f6ba39c-3267-49b9-a14b-53eeeb06678c


1. Demo: https://github.com/AnonNattOP/LeafScanWebApp/blob/441b91c56fb2c3c8054674ffcfc8218ff425509c/demo.mp4 
2. Easy: https://github.com/AnonNattOP/LeafScanWebApp/blob/441b91c56fb2c3c8054674ffcfc8218ff425509c/easy.mp4
3. Medium: https://github.com/AnonNattOP/LeafScanWebApp/blob/441b91c56fb2c3c8054674ffcfc8218ff425509c/medium.mp4
4. Hard: https://github.com/AnonNattOP/LeafScanWebApp/blob/441b91c56fb2c3c8054674ffcfc8218ff425509c/hard.mp4
