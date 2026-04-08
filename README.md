# SOHC-SASRec: Special Orthogonal Hyper-Connections for Sequential Recommendation

Official implementation of **SOHC-SASRec**, an advanced sequential recommendation model that leverages Special Orthogonal transformations to enhance signal propagation and interest modeling.

## 🚀 Overview

SOHC (Special Orthogonal Hyper-Connections) introduces a novel residual connection mechanism based on the **Cayley Transform** and **Newton-Schulz iteration**. By restricting the connection matrix to the Special Orthogonal group $SO(n)$, SOHC preserves the norm of the hidden states, effectively mitigating gradient vanishing/explosion issues and improving the modeling of dynamic interest drifts.

### Key Features
- **Isometric Rotation**: Maintains signal energy across layers.
- **Stable Gradient Flow**: High-performance training on deep architectures.
- **Seamless Integration**: Built on top of the robust [RecBole](https://recbole.io/) framework.

## 🛠️ Architecture

```ascii
                    +---------------------------+
                    |      SOHC Encoder Layer   |
                    +-------------+-------------+
                                  |
            +---------------------+---------------------+
            |                                           |
    [SASRec Attention]                        [Isometric Rotation]
            |                                           |
            |   SO(n) Projection via Cayley Transform   |
            |      R = (I - A)(I + A)^-1                |
            |                                           |
            +---------------------+---------------------+
                                  |
                    +-------------v-------------+
                    |      Synthesis & Output   |
                    +---------------------------+
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/findshan/sohc-sasrec.git
cd sohc-sasrec

# Install dependencies
pip install -r requirements.txt
```

## 📊 Experimental Results

| Dataset | Model | Recall@10 | NDCG@10 | MRR@10 |
| :--- | :--- | :--- | :--- | :--- |
| **MovieLens-100K** | SASRec | 0.1124 | 0.0545 | 0.0371 |
| | **SOHCSASRec** | **0.1273** | **0.0606** | **0.0407** |
| **Steam** | SASRec | 0.1205 | 0.0626 | 0.0453 |
| | **SOHCSASRec** | **0.1243** | **0.0642** | **0.0462** |
| **MovieLens-1M** | SASRec | 0.2642 | 0.1481 | 0.1122 |
| | **SOHCSASRec** | **0.2692** | **0.1511** | **0.1151** |

## 🧪 Quick Start

To verify the setup, run the demo experiment on ml-100k:

```bash
python run_test.py --dataset ml-100k --model SOHCSASRec
```

To run the full experimental suite:

```bash
python experiments/scripts/run_group.py --datasets ml-1m,steam,beauty,yelp --gpu 0
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{shan2026sohc,
  title={Special Orthogonal Hyper-Connections for Sequential Recommendation},
  author={Shan, Find and Liu, Wei},
  journal={arXiv preprint},
  year={2026}
}
```
