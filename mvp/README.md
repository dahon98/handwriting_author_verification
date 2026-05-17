# Handwriting Verification MVP

Web service for handwritten text author verification using CNN-CrossViT neural network.

## Quick Start

### 1. Clone repository
```bash
git clone https://github.com/dahon98/handwriting_author_verification.git
cd handwriting_author_verification/mvp
```

### 2. Download model
Download best_cnn_crossvit_c_b_240.pth from:
🔗 https://drive.google.com/file/d/1g2TcnBnNRtRx7n_RGT48ro5EpjvvXjAZ/view

Create models/ folder in handwriting_author_verification/mvp directory and place the model there.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
For CPU-only systems, use:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
### 4. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Usage
Upload Reference image (left column)

Upload Questioned image (right column)

Click "Verify Handwriting"

View decision and probability
