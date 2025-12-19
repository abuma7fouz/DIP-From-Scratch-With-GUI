# 👑 TEAM ELMOLOK 👑

**TEAM ELMOLOK** is a comprehensive **Digital Image Processing (DIP) toolkit built from scratch in Python**.
The project demonstrates core image processing concepts without relying on high-level built-in functions, making it ideal for **learning, teaching, and academic purposes**.

---

## 🚀 Features

TEAM ELMOLOK covers a wide range of image processing operations:

### 🖼️ Image Basics

* Load RGB & Grayscale images
* RGB to Grayscale conversion

### ✨ Intensity & Point Operations

* Brightness adjustment (Add, Subtract, Multiply, Divide)
* Image Complement
* Solarization

### ➕➖ Image Arithmetic

* Add two images
* Subtract two images (both directions)

### 📊 Histogram Processing

* Grayscale Histogram
* Histogram Stretching
* Histogram Equalization
* RGB Histograms
* RGB Histogram Stretching & Equalization

### 🧹 Spatial Filtering

* Mean Filter
* Median Filter
* Min / Max Filter
* Mode Filter
* Range Filter

### 🌫️ Smoothing & Sharpening

* Gaussian Smoothing (3x3 & custom size)
* Laplacian Filtering
* Custom Convolution Implementation

### 🎲 Noise Models

* Salt & Pepper Noise
* Gaussian Noise
* Periodic Noise

### 🧱 Morphological Operations

* Dilation
* Erosion
* Opening
* Closing
* All Morphological Operations Combined

### ✂️ Segmentation

* Otsu Automatic Thresholding

### 🖤 Dithering

* Floyd–Steinberg Dithering (1-bit)

---

## 🧠 Educational Goals

* Understand **Digital Image Processing fundamentals**
* Implement algorithms **pixel by pixel**
* Avoid black-box libraries when possible
* Visualize every step of the processing pipeline

This makes the project perfect for:

* Computer Vision students
* Image Processing courses
* Practical DIP labs

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy**
* **Matplotlib**
* **scikit-image**
* **Pillow (PIL)**

---

## 📂 Project Structure

```
TEAM-ELMOLOK/
│── app.py          # Main application entry (GUI / executable source)
│── code.py         # Core image processing functions & algorithms
│── win_app.rar     # Windows executable package (.exe)
│── README.md       # Project documentation
```

TEAM ELMOLOK/
│── code.py                # Main implementation
│── README.md              # Project documentation
│── images/                # Sample input images (optional)

````

---

## ▶️ How to Run

### 🔹 Option 1: Run as a Desktop Application (.exe)

The project is also available as a **Windows executable (.exe)**, allowing you to run it like a real desktop application **without installing Python or any dependencies**.

**How to use:**
1. Download the `.exe` file from the release folder.
2. Double-click the file.
3. The application will start and display image processing operations visually.

> This makes the project suitable for real-world usage, demos, and non-technical users.

---

### 🔹 Option 2: Run from Source Code (Python)


1. Clone the repository:
```bash
git clone https://github.com/your-username/TEAM ELMOLOK.git
cd TEAM ELMOLOK
````

2. Install dependencies:

```bash
pip install numpy matplotlib scikit-image pillow
```

3. Run the project:

```bash
python code.py
```

> Each function visualizes its output using Matplotlib.

---

## 📸 Sample Outputs

* Image enhancement
* Histogram transformations
* Noise & filtering comparisons
* Morphological transformations

(All results are displayed automatically during execution.)

---

## 📌 Notes

* All algorithms are implemented **from scratch** for clarity.
* The project prioritizes **readability and learning** over performance.
* Ideal as a **reference or starting point** for advanced CV projects.

---

## 🤝 Contribution

Contributions are welcome!

* Improve performance
* Add new filters
* Refactor code
* Extend to frequency domain processing

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it for learning and research.

---

## 👨‍💻 Author

Developed with ❤️ for learning Digital Image Processing.

> *TEAM ELMOLOK — where pixels turn into insight.* ✨
