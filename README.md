# Human Immune Multiomics Atlas - Pipeline v2

![Status](https://img.shields.io/badge/Status-Completed-success)
![Pipeline](https://img.shields.io/badge/Pipeline-v2-blue)

## Overview
This repository contains Pipeline v2 for the Human Immune Multiomics Atlas. It leverages a Multimodal Variational Autoencoder (MVAE2) to effectively integrate single-cell RNA and ATAC-seq data into a shared latent space. By combining these modalities, the pipeline achieves highly accurate predictions and classification across diverse human immune cell populations.

## Key features
* **Multimodal integration:** Joint analysis of scRNA (UMAP) and ATAC-seq (Fragments per Cell vs. Peaks) using an advanced MVAE2 architecture.
* **Latent Space Analysis:** Robust cell type clustering and visualization within a shared latent space.
* **High accuracy:** Exceptional classification accuracy across both training and validation phases.

## Model performance
The MVAE2 model demonstrates robust training and validation metrics, achieving a peak validation accuracy of approximately 0.98 over 120 epochs. 

Based on the predicted vs. true confusion matrix, the pipeline successfully classifies major immune cell populations with the following high-confidence scores:

* **B naive:** 1.00 
* **Plasmacytoid Dendritic Cells (PDC):** 1.00 
* **CD16+ Monocytes:** 0.99 
* **CD4+ T naive:** 0.98 
* **CD14+ Monocytes:** 0.97 
* **Natural Killer (NK) cells:** 0.97 
* **CD8+ T cells:** 0.96 
* **Dendritic Cells (DC):** 0.96 
* **CD4+ T memory:** 0.95

Additionally, the pipeline tracks per-batch AUROC and Pearson correlations for comprehensive cross-modality (ATAC-RNA) evaluation.
