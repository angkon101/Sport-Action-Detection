# Research Paper Writing Workspace: CMAT-HOI

> **Project Title**: CMAT-HOI: Cross-Modal Attention Transformer for Human–Object Interaction in Fine-Grained Sports Action Recognition  
> **Dataset**: 82 Sports Categories, 9,151 Videos (UCF Sports Action Benchmark)  
> **Key Metric Achieved**: **90.19% Top-1 Accuracy** | **99.02% Top-5 Accuracy** | **0.4397 Test Loss**  
> **Workspace Directory**: `E:\Akhi Thesis\Paper writing`

---

## 📁 Workspace Directory Structure

```text
Paper writing/
├── figures/                              # High-resolution (300 DPI) publication-ready figures
│   ├── system_architecture_diagram.png   # CMAT-HOI Block Diagram (Stems, Cross-Attn, GMFU, Transformer)
│   ├── dataset_split_distribution.png    # 82-class train/test sample distribution
│   ├── loss_curves.png                   # Train vs. validation loss convergence
│   ├── accuracy_curves.png               # Top-1 and Top-5 accuracy progression over 45 epochs
│   ├── confusion_matrix_82.png           # 82x82 normalized confusion matrix
│   ├── per_class_accuracy.png            # Ranked per-class performance bar chart
│   ├── temporal_attention_weights.png    # Frame-level cross-attention weight heatmaps
│   ├── metric_distribution_82.png        # Boxplot distribution of F1, Precision, and Recall
│   └── multimodal_ablation_comparison.png# M1 to M7 multimodal ablation study chart
├── references.bib                        # Comprehensive BibTeX citations file
├── PAPER_OUTLINE_AND_DRAFT.md            # Complete structured manuscript draft with equations & tables
└── main.tex                              # LaTeX manuscript source (ready for Overleaf / IEEE / Springer)
```

---

## 🎯 Paper Core Information & Contributions

### 1. Working Title
**CMAT-HOI: Cross-Modal Attention Transformer for Human–Object Interaction in Fine-Grained Sports Action Recognition**

### 2. Primary Scientific Contributions
1. **Modality Representation Alignment**: Solves representation collapse where 2048-d visual features overwhelm 85-d skeleton and 7-d object streams by projecting all modalities into uniform $d=256$ representation spaces using 1D temporal convolutions and LayerNorm.
2. **Bi-Directional Keypoint–Equipment Cross-Attention**: Introduces mutual attention query-key-value mechanisms ($Q_P \leftrightarrow K_O, V_O$ and $Q_O \leftrightarrow K_P, V_P$) between skeletal joints and sports apparatus, modeling fine-grained physical dynamics (e.g., bat angle, racquet grip, ball trajectory).
3. **Gated Multimodal Fusion Unit (GMFU)**: Dynamically assigns soft frame-level gating weights to each stream, effectively downweighting cluttered scene backgrounds when kinematic cues are strong.
4. **Multi-Scale 4-Way Temporal Pooling**: Concatenates CLS token, temporal attention pooling, max pooling, and average pooling ($1024$-d) to capture instantaneous explosive movements alongside sustained temporal trends.
5. **Rigorous Validation Across 82 Categories**: Achieved **90.19% Top-1** and **99.02% Top-5** accuracy on 2,743 disjoint test videos, outperforming the legacy BiLSTM baseline (55.20%) by **+34.99%**.

---

## 📝 Roadmap for Paper Writing
1. **Structure & Abstract**: Finalize the core narrative, motivation, and abstract.
2. **Methodology & Mathematical Rigor**: Detail the tensor equations, attention formulation, and GMFU gating mechanics.
3. **Experimental Results & Tables**: Transcribe exact epoch logs, confusion matrix insights, and M1-M7 ablation tables.
4. **Target Venue Selection**: Align formatting with IEEE (IEEE Transactions on Multimedia / IEEE Access), Elsevier (Pattern Recognition), Springer (MTAP), or standard conference template.
