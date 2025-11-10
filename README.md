# GPThash

> **Research Project: Long-Term Airborne Trajectory Sequence Prediction Based on Large Language Models**  
> Undergraduate Innovation Training Program, University of Electronic Science and Technology of China (UESTC)  
> Project No.: X202510614120 | Duration: Dec 2024 – Oct 2025  

## Overview

This project investigates the application of large-model paradigms—particularly Transformer-based architectures—to the task of **long-term airborne trajectory prediction**. Inspired by advances in natural language processing, we propose a novel **"trajectory-as-language"** modeling framework, where continuous flight trajectories are discretized into token-like sequences and processed using self-attention mechanisms.

The goal is to overcome the limitations of traditional recurrent models (e.g., LSTM) in capturing long-range spatiotemporal dependencies, thereby enabling more accurate, stable, and interpretable predictions for air traffic management, flight safety assurance, and anomaly detection.

## Key Innovations

- **Trajectory Language Modeling**: Flight coordinates (latitude, longitude, altitude) are discretized into symbolic tokens using an enhanced spatial encoding scheme, allowing trajectory sequences to be treated analogously to natural language.
  
- **Velocity Prompt Tokens**: We introduce a **velocity-aware prompting mechanism**, where recent velocity vectors (computed via finite differences over the last 10 time steps) are quantized and embedded as contextual prompts. This significantly enriches the model’s understanding of motion dynamics.

- **Geohash3 Encoding with Altitude Integration**: An extended Geohash variant (**Geohash3**) incorporates altitude alongside horizontal coordinates, enabling unified 3D spatial discretization while preserving locality.

- **Positional Encoding Optimization**: Custom positional encodings are designed to better reflect the temporal and spatial structure of airborne trajectories, improving convergence and generalization.

## Usage Instructions

To reproduce our pipeline:

1. **Build the tokenizer vocabulary**:  
   Run `train_token.py` to generate the trajectory token dictionary (e.g., `tokenizer_3D_7+1word_blur.json`).  
   ⚠️ Note: The output path (`save_path`) must be manually set inside `train_token.py` before execution.

2. **Train the prediction model**:  
   After obtaining the tokenizer file, run `train.py` to train the Transformer-based trajectory prediction model. Ensure the tokenizer path is correctly specified in the training configuration.

## Related Work & Inspiration

This approach draws direct inspiration from [**trAISformer** (*"TrAISformer -- A Transformer Network with Sparse Augmented Data Representation and Cross Entropy Loss for AIS-based Vessel Trajectory Prediction"*, arXiv 2024)](https://arxiv.org/abs/2109.03958), which demonstrated the feasibility of applying Transformer architectures to maritime vessel trajectory forecasting via tokenization of Automatic Identification System (AIS) data. Building on this foundation, this work extends the paradigm to **3D aerial domains**, introduces **velocity prompting**, and enhances spatial encoding for aviation-specific challenges.

## Acknowledgements

We gratefully acknowledge the guidance of **Prof. Jing Liang** from the School of Information and Communication Engineering, UESTC. This project was supported by the **Undergraduate Innovation Training Program of UESTC**.

## Author
**Xi-Yao Chen** – Glasgow College, UESTC (BEng in Communication Engineering, Joint Program)

> ✈️ *Enabling intelligent airspace through sequence modeling beyond language.*
