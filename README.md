# SOHC-SASRec: Special Orthogonal Hyper-Connections for Sequential Recommendation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![RecBole](https://img.shields.io/badge/RecBole-1.1.1+-green.svg)](https://recbole.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **SOHC-SASRec**, an advanced sequential recommendation model that leverages Special Orthogonal transformations to enhance signal propagation and interest modeling.

---

## 🚀 Introduction

Sequential recommendation aims to predict the next item based on user historical interactions. While self-attention mechanisms (e.g., SASRec) have achieved SOTA performance, they often suffer from gradient vanishing or signal dilution in deep architectures.

**SOHC (Special Orthogonal Hyper-Connections)** introduces a novel residual connection mechanism that ensures **Isometric Rotation** of hidden states. By constraining the connection matrix to the Special Orthogonal group $SO(n)$, SOHC preserves the energy of the signal across layers, leading to more stable training and better capture of long-term dependencies.

### 🌟 Key Concepts

1.  **Isometric Signal Preservation**: Unlike standard residual connections ($x + f(x)$), SOHC uses a rotation-based synthesis that maintains $\|x\|_2$.
2.  **Cayley Transform Engine**: We map skew-symmetric matrices to the orthogonal manifold using the Cayley transform:
    $$R = (I - A)(I + A)^{-1}$$
3.  **Newton-Schulz Gradient Smoothing**: We utilize Newton-Schulz iteration to efficiently approximate the matrix inverse, ensuring smooth backpropagation and high throughput.

---

## 🛠️ Architecture

```ascii
                    +------------------------------------+
                    |         SOHC Encoder Layer         |
                    +-----------------+------------------+
                                      |
            +-------------------------+-------------------------+
            |                                                   |
    [SASRec Self-Attention]                         [Special Orthogonal Rotation]
            |                                                   |
            |           +---------------------------+           |
            |           |  Cayley Transform Engine   |           |
            |           |  R = (I - A)(I + A)^-1    |           |
            |           +-------------+-------------+           |
            |                         |                         |
            |           +-------------v-------------+           |
            |           |  Newton-Schulz Iteration  |           |
            +-----------+  X_{k+1} = X_k(3I - X_k^2)/2 +-----------+
                        +-------------+-------------+
                                      |
                        +-------------v-------------+
                        |     Final Gated Synthesis  |
                        +---------------------------+
```


## 📦 Installation & Quick Start

### 1. Requirements
Ensure you have Python 3.8+ and PyTorch installed.
```bash
git clone https://github.com/findshan/sohc.git
cd sohc
pip install -r requirements.txt
```

### 2. Verify with Demo (ml-100k)
Run a quick comparison run between Baseline and SOHC on the MovieLens-100K dataset:
```bash
python compare_models.py --gpu 0 --epochs 50
```

### 3. Run Full Experiments
To replicate results on larger datasets:
```bash
python experiments/scripts/run_group.py --datasets ml-1m,steam,beauty,yelp --gpu 0
```

---

## 📝 Citation

This work is published at **ACM CIKM 2026** (November 7-11, 2026, Rome, Italy).

```bibtex
@inproceedings{deng2026sohc,
  title     = {Special Orthogonal Hyper-Connections for Geometry-Preserving Deep Routing},
  author    = {Liu, Wei and Zhang, Xinyi and Deng, Wenshan and Zhu, Huaijie and
               Zheng, Libin and Yu, Jianxing and Yin, Jian},
  booktitle = {Proceedings of the 35th ACM International Conference on
               Information and Knowledge Management (CIKM '26)},
  year      = {2026},
  publisher = {Association for Computing Machinery}
}
```

---

## 👤 Contributions

**Method design, implementation, experiments and paper writing: Wenshan Deng** ([@findshan](https://github.com/findshan))

- Proposed the SOHC framework: exact isometric routing on the special orthogonal group via the Cayley transform.
- Designed and implemented the gated dual-stream architecture (isometric propagation + semantic refinement).
- Ran every experiment reported in the paper:
  - **36 / 60-layer NanoGPT** on FineWeb10B (validation loss, training-time Pareto frontier, depth sweep)
  - **12-layer SASRec** on ML-1M / ML-100K (MRR@10, HR@10, NDCG@10)
  - **Four-way ablations** isolating orthogonal constraint / semantic stream / routing dimension / adaptive gate
  - **Effective-rank analysis** of representation degeneration
- Wrote the manuscript and the code in this repository.

**Timeline note.** The method, all code and every experiment in this repository predate the paper's
camera-ready submission; the commit history is the authoritative record of when each piece was produced.

## 🙏 Acknowledgement
This project is built upon the excellent [RecBole](https://github.com/RUCAIBox/RecBole) framework.
