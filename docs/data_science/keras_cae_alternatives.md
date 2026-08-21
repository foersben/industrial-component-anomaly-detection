---
type: Data Science
title: "Classical Alternatives & Design Decisions"
description: "Technical documentation for Classical Alternatives & Design Decisions."
tags: [keras, autoencoder, documentation]
---

# Classical Alternatives & Design Decisions

This page explains the rationale behind the architectural choices made in the Keras Convolutional Autoencoder (CAE) for anomaly detection, specifically comparing them to popular alternatives.

## Activation Functions: Why ELU?

The standard Rectified Linear Unit (ReLU) has a well-known failure mode: the **"Dying ReLU"** problem. If a neuron receives consistently negative inputs during training, its gradient becomes permanently zero, and the neuron stops contributing to learning entirely. This is especially problematic in autoencoders where the bottleneck severely constrains information flow.

The **Exponential Linear Unit (ELU)** smoothly saturates for large negative inputs instead of zeroing them:

- For `x > 0`: `ELU(x) = x` (same as ReLU)
- For `x <= 0`: `ELU(x) = alpha * (exp(x) - 1)` (where `alpha` is typically 1.0)

### Why not Leaky ReLU?

While Leaky ReLU also prevents dying neurons by adding a small linear slope for negative inputs, it has a sharp kink at `x = 0` (it is not continuously differentiable). This sharp non-linearity can sometimes destabilize the fine-grained reconstruction gradients in autoencoders.

ELU is smooth everywhere, producing more predictable gradients. Furthermore, ELU's saturation curve naturally pushes mean activations closer to zero (a "self-normalizing" property), which speeds up learning and acts like internal batch normalization - a benefit Leaky ReLU lacks.

### Why not Gated Linear Units (GLUs)?

GLUs (and variants like SwiGLU) are extremely powerful, particularly in Transformers, because they provide a learned, data-dependent gating mechanism. However, they require two parallel projections (one for the gate, one for the value) per layer.

This effectively doubles the parameter count, compute requirements, and memory footprint of the activation step. In a purely convolutional autoencoder designed for anomaly detection - especially one running under strict memory constraints - the overhead of GLUs is prohibitive. ELU provides the necessary representational power for spatial feature maps at a fraction of the computational cost.

## Summary of ELU Benefits

- **Neurons never completely die** -> stable gradient flow throughout training.
- **Mean activations closer to zero** -> network acts like batch normalisation internally.
- **Smooth gradients everywhere** -> better fine-grained structural reconstruction.
