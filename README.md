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

The proposed **GeoMeta-GT** framework consists of the following stages:

1. **Frozen CNN Feature Extraction:** Extract spatially structured deep feature maps from dermoscopic images using a pretrained ResNet152 backbone, which remains frozen during training to reduce computational overhead and the number of trainable parameters.

2. **Deep Feature-Space Superpixel Decomposition:** Apply Simple Linear Iterative Clustering (SLIC) to the upsampled deep feature maps to partition each image into spatially coherent regions. This feature-space segmentation aims to capture semantically meaningful lesion structures.

3. **Region-Level Feature Representation:** Represent each superpixel as a graph node by concatenating mean-pooled and max-pooled deep features, combining global regional context with salient local information.

4. **Geometry-Aware Graph Construction:** Connect superpixel nodes using nearest-neighbour relationships based on their spatial coordinates. Encode inter-region Euclidean distance and relative orientation as edge attributes to preserve the geometric structure of the lesion.

5. **Metadata-as-Node Fusion:** Encode clinical metadata, including demographic and anatomical information, into a dedicated metadata node. Connect this node bidirectionally to all superpixel nodes, enabling joint visual and clinical reasoning through graph message passing rather than conventional late fusion.

6. **Edge-Aware Graph Transformer:** Integrate geometric edge attributes directly into the attention mechanism and message aggregation process, allowing the model to adaptively weight neighbouring regions according to their visual representations and spatial relationships.

7. **Similarity-Weighted Structural Refinement:** Refine the updated node embeddings through similarity-weighted propagation, encouraging semantically consistent regions to exchange information while suppressing inconsistent or noisy representations.

8. **Graph-Level Representation and Classification:** Apply global mean pooling to aggregate the refined node embeddings into a graph-level representation, followed by a trainable classifier to predict benign or malignant lesions.

<p align="center">
  <img src="src/model_fig.jpg" width="800" alt="Context-Aware GNN Architecture Diagram"><br>
  <em>Figure: GeoMeta-GT combines frozen deep visual representations, geometry-aware region-level graph reasoning, and graph-native clinical metadata integration within a parameter-efficient classification framework.</em>
</p>

---










