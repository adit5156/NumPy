# 📊 NumPy – Foundations for Data Analytics

<p align="center">
  <b>Building a Strong Numerical Computing Foundation for Data Analytics & Machine Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/Library-NumPy-orange">
  <img src="https://img.shields.io/badge/Level-Beginner%20to%20Intermediate-green">
  <img src="https://img.shields.io/badge/Status-Completed-success">
  <img src="https://img.shields.io/badge/License-Educational-lightgrey">
</p>

---

## 📌 Introduction

**NumPy (Numerical Python)** is the foundational library for numerical and scientific computing in Python.

If you are moving toward **Data Analytics, Data Science, or Machine Learning**, mastering NumPy is essential because major libraries such as:

- Pandas  
- SciPy  
- Scikit-learn  
- TensorFlow  

are built on top of NumPy arrays.

---

## 🚀 What NumPy Provides

- ✅ Fast multidimensional arrays (`ndarray`)
- ✅ Vectorized operations (eliminates explicit loops)
- ✅ Mathematical & statistical functions
- ✅ Linear algebra support
- ✅ Efficient memory utilization
- ✅ High performance compared to Python lists

---

## ❓ Why NumPy is Important

### 🔹 In Standard Python:

- Lists are slower for mathematical operations  
- No native vectorized computation  
- Inefficient for large-scale numerical data  
- Higher memory overhead  

### 🔹 With NumPy:

- Operations optimized in C (faster execution)
- Supports multidimensional array structures
- Enables broadcasting
- Built-in advanced mathematical operations
- Better memory efficiency

---

# ⚙️ Installation

### 🔹 Install using pip

```bash
pip install numpy
```

### 🔹 Install using conda

```bash
conda install numpy
```

### 🔹 Verify Installation

```python
import numpy as np
print(np.__version__)
```

---

# 📦 Importing NumPy

```python
import numpy as np
```

---

# 📊 Working with NumPy Arrays

---

## 1️⃣ Creating Arrays

### 🔹 1D Array

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])

print(arr1)
print(type(arr1))
print(arr1.shape)

print(arr1.reshape(1, 5))
print(arr1.reshape(5, 1))
```

---

### 🔹 2D Array

```python
arr2 = np.array([[1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 6]])

print(arr2)
print(arr2.shape)
```

---

## 2️⃣ Array Creation Functions

### 🔹 Using `arange()`

```python
np.arange(0, 10, 2).reshape(1, 5)
```

### 🔹 Using `ones()`

```python
np.ones((3, 2))
```

### 🔹 Identity Matrix

```python
np.eye(3)
```

---

## 3️⃣ Properties in NumPy

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("Array:\n", arr)
print("Shape:", arr.shape)
print("Number of dimensions:", arr.ndim)
print("Size (number of elements):", arr.size)
print("Data type:", arr.dtype)
print("Item size (in bytes):", arr.itemsize)
```

---

## 🔎 Key Attributes Explained

| Attribute  | Description |
|------------|------------|
| `shape`    | Dimensions of the array |
| `ndim`     | Number of axes (dimensions) |
| `size`     | Total number of elements |
| `dtype`    | Data type of elements |
| `itemsize` | Memory used by each element (in bytes) |

---

# 🧠 Core Concepts Covered

- Array creation (1D & 2D)
- Reshaping arrays
- Built-in array generators (`arange`, `ones`, `eye`)
- Understanding dimensions
- Array metadata & memory structure
- Foundation for data preprocessing

---

# 🎯 Learning Outcomes

After completing this module, you will:

- Understand NumPy array fundamentals
- Work confidently with multidimensional arrays
- Inspect and interpret array properties
- Perform efficient numerical operations
- Build a strong base for Data Analytics workflows

---

# 📂 Repository Structure

```
Numpy.ipynb
README.md
```

---

# 📈 Future Enhancements (Roadmap)

- [ ] Broadcasting examples
- [ ] Vectorized arithmetic operations
- [ ] Performance benchmark (Python list vs NumPy)
- [ ] Linear algebra operations
- [ ] Mini analytics exercises
- [ ] Real dataset practice examples

---

# 🧑‍💻 Who This Repository Is For

- Beginners starting Data Analytics
- Python learners transitioning into numerical computing
- Students preparing for Data Science roles
- Anyone building a strong NumPy foundation

---

# 🤝 Connect & Collaboration

This repository is part of my structured journey toward becoming a **Data Analyst**.  
I am continuously building and documenting foundational modules in Python, NumPy, Pandas, and Data Visualization.

If you are a recruiter, mentor, or fellow learner — feedback and collaboration are welcome.

---

# 📄 License

This project is created for educational and portfolio purposes.
