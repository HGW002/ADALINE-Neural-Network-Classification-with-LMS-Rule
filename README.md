# ADALINE Neural Network Classification with LMS Rule

This project implements an **ADALINE (Adaptive Linear Neuron)** neural network to solve a **multi-class classification problem** using the **Least Mean Squares (LMS)** learning rule. The problem is based on
> *Neural Network Design* by Martin T. Hagan, Howard B. Demuth, and Mark H. Beale

---

## Problem Summary

We are given 8 input vectors, grouped into 4 distinct classes:

- **Class 1**: [1, 1], [1, 2] → Target: [-1, -1]  
- **Class 2**: [2, -1], [2, 0] → Target: [-1, 1]  
- **Class 3**: [-1, 2], [-2, 1] → Target: [1, -1]  
- **Class 4**: [-1, -1], [-2, -2] → Target: [1, 1]

Each input vector is 2-dimensional, and each class is mapped to a unique 2D target vector.

The ADALINE network has **two output neurons**, and it is trained using the **LMS (delta) rule** to minimize the error between actual and target outputs.

------------------------------------

## Key Features

- Implements a **2-output ADALINE** using the **Least Mean Squares (LMS)** learning algorithm
- Handles **4-class classification**
- Visualizes **decision boundaries** in 2D input space

