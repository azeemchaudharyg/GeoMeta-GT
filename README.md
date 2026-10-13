# Geometry-Aware Superpixel Graph Transformer with Metadata for Skin Lesion Classification

<p align="left">
  <a href="#" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/MICCAI-2026-blue" alt="Conference"></a>
  <a href="https://doi.org/10.1007/978-3-032-38091-3_31" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/DOI-10.1007/978.3.032.38091.3..31-red" alt="DOI"></a>
  <a href="https://github.com/azeemchaudharyg/GeoMeta-GT/blob/main/train.py" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Framework-PyTorch_Geometric-purple" alt="Implementation"></a>
  <a href="#" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

Official repository accompanying the paper: **"Geometry-Aware Superpixel Graph Transformer with Metadata for Skin Lesion Classification"** 

**Authors:** Muhammad Azeem, Tanveer Hussain, Amr Ahmed, and Ardhendu Behera  

**Institution:** Edge Hill University, Ormskirk, Lancashire, UK

---

## Abstract
Automated skin cancer classification from dermoscopic images remains challenging due to heterogeneous lesion structure, strong intra-class variability, and subtle visual differences between benign and malignant cases. Existing CNN/ViT pipelines typically rely on global or patch-level features and often combine patient metadata via late fusion, which limits spatially grounded multimodal reasoning. We present a novel region-based graph learning framework that explicitly models lesions as graphs of spatially coherent superpixel regions represented as frozen CNN features. To capture fine-grained lesion arrangements, we encode inter-regional geometry as edge attributes and introduce a dedicated metadata context node connected to all regions, providing structured integration of demographic/clinical variables within the same relational space. Node representations are updated using our edge-aware graph transformer followed by attention-driven propagation, and a final graph-level embedding for benign–malignant classification. Experiments on four public benchmarks demonstrate that explicit region-level relational modeling and graph-native multimodal fusion yield consistent gains over the state-of-the-art. Consequently, we establish a new graph-centric perspective in which CNN features are modeled as relational nodes and improved through contextual integration, yielding more expressive and robust classifications.

---


## Key Contributions

* **Parameter-Efficient Classification Framework:** A spatially grounded and multimodally structured framework for robust dermoscopic skin lesion classification using a frozen backbone.

* **Geometry-Attributed Superpixel Graph Representation:** A region-level graph representation that captures spatial relationships between superpixels and integrates clinical metadata through a dedicated metadata node.

* **Edge-Aware Graph Transformer:** A graph transformer that leverages geometric edge attributes to facilitate spatially informed message passing and diagnostically relevant region aggregation.

* **Similarity-Weighted Region Refinement:** A refinement module that enhances feature consistency across semantically similar regions, improving the quality of graph representations.

* **Comprehensive Multi-Benchmark Evaluation:** Consistent performance improvements across multiple public skin lesion classification benchmarks, demonstrating the effectiveness of the proposed framework while retaining a frozen feature extraction backbone.

---

## Method Overview
<p align="center">
  <img src="src/model_fig.jpg" width="800" alt="Context-Aware GNN Architecture Diagram"><br>
  <em>Figure: Overview of the Context-Aware Graph Neural Network pipeline for lesion classification.</em>
</p>

1. **Graph Processing Pipeline:** The joint multimodal feature vector $F \in \mathbb{R}^D$ is partitioned into $N$ unique subspaces (e.g., $N=6$) representing nodes in a fully connected graph, which undergoes relation-aware feature aggregation and multi-hop message passing to map long-range dependencies.
2. **Downstream Classification:** The updated, context-rich graph nodes are aggregated via global mean pooling to construct a singular, graph-level feature embedding that is passed directly to a classifier to yield the final diagnostic prediction (benign vs. malignant).
3. **Ablation Framework:** To isolate multimodal performance impacts, an alternative late-fusion architecture variant is developed where metadata tracks are separately encoded through a Multi-Layer Perceptron (MLP) and merged with graph-processed image nodes right before the final classification head.

---










