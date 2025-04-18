# Physics-Constrained Convolutional Recurrent Neural Network (PhyCRNet)

This repository provides the official implementation of **PhyCRNet**, a physics-constrained convolutional recurrent neural network tailored to solve spatial-temporal partial differential equations (PDEs) with **arbitrary boundary conditions**. It is designed for scientific machine learning applications where labeled data is scarce or unavailable.

PhyCRNet extends traditional physics-constrained neural networks by introducing:

- **Boundary condition handling as soft constraints** via finite-difference-based differentiation
- **Adaptive loss weighting** to improve convergence and training efficiency
- **Convolutional + recurrent architecture** to model spatial-temporal dynamics

If you use this code in your research, please cite:

> Li, G., Lu, Y., & Liu, D. (2025). Solving spatial-temporal PDEs with arbitrary boundary conditions using physics-constrained convolutional recurrent neural networks. *Neurocomputing, 634*, 129917. https://doi.org/10.1016/j.neucom.2024.129917

---

## 🧠 Features

- Physics-constrained learning (no labeled data needed)
- Handles mixed and arbitrary boundary conditions
- Applies to nonlinear PDEs including:
  - Allen–Cahn equation
  - 2D Burgers’ equation
  - Reaction-diffusion systems
- Designed using PyTorch and ConvLSTM architecture



