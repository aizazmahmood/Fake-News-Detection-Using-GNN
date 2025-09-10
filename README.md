# Fake-News-Detection-Using-GNN

A research-grade pipeline for **fake-news detection** that fuses a strong **text branch (MPNet)** with a **heterogeneous GNN branch (GraphSAGE/HeteroSAGE)**, combined via **posterior-level ensembling** for robust, class-balanced performance on **four-way veracity** labels (mostly true, mixture, mostly false, no factual content). The design is intentionally pragmatic: *simple to reproduce on student hardware, yet competitive and transparent.* :contentReference[oaicite:0]{index=0}

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Installing PyTorch Geometric](#installing-pytorch-geometric)
- [Datasets & Layout](#datasets--layout)
- [Method Overview](#method-overview)
- [Why This Design](#why-this-design)
- [Reproducibility & Commands](#reproducibility--commands)
- [Results (Dev/Test) & α-Sweep](#results-devtest--α-sweep)
- [Ablations](#ablations)
- [Troubleshooting](#troubleshooting)
- [Ethics, Risk, and Responsible Use](#ethics-risk-and-responsible-use)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights

- **Hybrid detector:** MPNet text classifier + HeteroSAGE over a typed graph; **late fusion** of softmax posteriors.  
- **Realistic 4-class setup:** mostly true / mixture / mostly false / no factual content; macro-F1 focused evaluation.
- **Imbalance-aware GNN:** **focal loss (γ≈3.0)** plus class-weighting boosts minority recall. 
- **Pragmatic engineering:** small, reproducible heterogeneous graph (Articles, Domains, Hosts, Entities) that trains on commodity GPUs/CPU.
- **Posterior-level ensemble:** single weight **α**; simple sweep yields consistent, calibrated gains in macro-F1.

---

## Repository Structure
- data/ # raw/ and processed/ datasets, graphs, splits
- models/ # saved weights/checkpoints
- outputs/ # metrics, confusion matrices, plots, reports
- src/ # code (data prep, graph build, train, eval)
- requirements.txt # Python dependencies

This reflects the current repo layout on GitHub. :contentReference[oaicite:6]{index=6}

---

## Quick Start

> **Python 3.10+** recommended. Use a fresh virtual environment.

# 1) Clone
git clone https://github.com/aizazmahmood/Fake-News-Detection-Using-GNN.git
cd Fake-News-Detection-Using-GNN

# 2) Environment
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1

# 3) Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Installing PyTorch Geometric

PyG uses platform-specific wheels. Pick the command for your PyTorch/CUDA setup from the official index. Example for CPU-only:
pip install torch-geometric torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.3.0+cpu.html

# Datasets & Layout

Supported: BuzzFeed (FakeNewsNet), and extendable to PolitiFact / GossipCop.
data/
  raw/
    buzzfeed/         # raw dumps/xml/csv/json
    politifact/
    gossipcop/
    fakenewsnet/
  interim/
    text/             # cleaned text
    entities/         # NER outputs
    links/            # hyperlink tables
    topics/           # topic features/embeddings (optional)
  processed/
    graphs/
      buzzfeed/
        hetero_graph.pt   # torch_geometric HeteroData
        splits.json       # train/dev/test indices
  metadata/
    domains.csv
    entity2id.json
    topic2id.json

Graph schema (final runs): Article, Domain, Host, Entity nodes; edges for belongs_to, links_to, has_entity (article-similarity edges were pruned for stability)

# Method Overview

Two branches + late fusion:

Text branch: MPNet fine-tuned for 4-way classification → posterior 𝑝𝑡𝑒𝑥𝑡(𝑦∣𝑥).

GNN branch: Heterogeneous GraphSAGE with typed aggregators and focal loss → posterior pgnn (y∣x, G).

Ensemble: pens=α ptext + (1−α)pgnn, tune α on dev (macro-F1). 


Design goals: complementarity (semantics + relations), reproducibility, and calibration.

# Why This Design
Operational constraints (early classification, topic drift, scarce expert labels) favor:

- a strong text encoder for lexical generalization,

- a compact hetero-graph to inject source/neighbor context without deep trees, and

- posterior-level fusion (simpler to calibrate/debug than feature fusion). 

Engineering trade-offs:

- HeteroSAGE > HAN here due to PyG API/extension stability on Windows; preserves most of the benefit. 

- Prune article-similarity edges to reduce noise/memory; improves generalization.

# Reproducibility & Commands
All commands below reflect the pipeline used to produce the reported numbers.

## Build graph (BuzzFeed)
Build heterogeneous graph with entities
python -m src.graphs.build_buzzfeed_graph \
  --buzzfeed_csv data/processed/buzzfeed_all.csv \
  --split_dir   data/processed/buzzfeed_split \
  --out_path    data/processed/graphs/buzzfeed_hetero.pt \
  --with_entities

 Prune article–article similarity edges (final runs)
python -m src.graphs.prune_edges \
  --in_pt  data/processed/graphs/buzzfeed_hetero.pt \
  --out_pt data/processed/graphs/buzzfeed_no_sim.pt \
  --drop "('article','similar','article')"

## Train branches
MPNet (text branch) – final checkpoint used in ensembles
outputs/hf/buzzfeed_mpnet/checkpoint-432
python -m src.models.train_hf_classifier_fnn  ...  # (see src/ for args)

HeteroSAGE (graph branch) with focal loss (γ=3.0), sum aggregator
python -m src.models.train_hetero_gnn \
  --graph_path data/processed/graphs/buzzfeed_no_sim.pt \
  --out_dir outputs/gnn/bf_hetero_sage_no_sim_focal(3.0) \
  --hidden 64 --dropout 0.3 --lr 0.002 --epochs 200 \
  --patience 20 --aggr sum --focal_gamma 3.0 --cw_pow 0.5

## Make ensembles & score
Fuse branch posteriors with alpha (α)
python -m src.models.ensemble_softmax \
  --buzzfeed_csv data/processed/buzzfeed_all.csv \
  --model_dir_hf outputs/hf/buzzfeed_mpnet/checkpoint-432 \
  --graph_pt data/processed/graphs/buzzfeed_no_sim.pt \
  --gnn_weights outputs/gnn/bf_hetero_sage_no_sim_focal(3.0)/heterosage_best.pt \
  --split_csv data/processed/buzzfeed_split/test.csv \
  --out_csv outputs/ensembles/bf_test_probs_a0.70.csv --alpha 0.70

python -m src.eval.score_probs \
  --split_csv data/processed/buzzfeed_split/test.csv \
  --probs_csv outputs/ensembles/bf_test_probs_a0.70.csv \
  --hf_model_dir outputs/hf/buzzfeed_mpnet/checkpoint-432 \
  --out_dir outputs/ensembles/sweep/test_a0.70

# Results (Dev/Test) & α-Sweep
- Single-branch GNN (HeteroSAGE, no-sim, focal γ=3.0):
Dev macro-F1 ≈ 0.503, Test macro-F1 ≈ 0.508, Test Acc ≈ 0.759. 

- Ensemble (dev-selected α = 0.75):
Dev macro-F1 ≈ 0.579, Test macro-F1 ≈ 0.520. 


- Ensemble (α = 0.70, strongest Test macro-F1):
Test macro-F1 ≈ 0.531, Acc ≈ 0.767. 


- α-sweep (dev macro-F1): peaks near α ∈ [0.70, 0.75], confirming complementarity and mild tilt toward text branch. 


As a reference, a prior BERT/MPNet text baseline alone achieved ~92% accuracy (split-dependent); the ensemble improves class balance (macro-F1), especially on minority labels. 

# Ablations
- Remove article–similarity edges: smaller memory, better generalization; dev may rise with sim-edges, but test macro-F1 drops—final runs prune them. 

- Aggregator: sum > mean on this sparse heterograph. 

- Loss: focal (γ=3.0) > cross-entropy for minority recall.

# Troubleshooting
- PyG wheels on Windows (scatter/sparse/cluster) must match PyTorch/CUDA exactly. 
- HANConv.__init__() got multiple values for 'metadata' → PyG API shift; remove extra kwarg or upgrade. 
- mat1/mat2 dtype mismatch → force .float() features before message passing. 
- Typed ParameterDict → index by string keys, e.g., self.lins['article__belongs_to__domain']. 
- Pandas join KeyError (id dtype) → keep ids as strings end-to-end.
- Unicode on Windows → write files with encoding='utf-8'.

# License
MIT (recommended).

# Acknowledgements
Thanks to the PyTorch Geometric community and the maintainers of BuzzFeed/FakeNewsNet.

