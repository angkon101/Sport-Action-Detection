# Related Papers & Literature Review Package for CMAT-HOI

This folder contains a complete, scholarly research package specifically compiled for writing the **Related Work** section of your research paper / master's thesis on **CMAT-HOI: Cross-Modal Attention Transformer for Human–Object Interaction in Fine-Grained Sports Action Recognition**.

---

## 📂 File Directory & Overview

| File | Description | Recommended Usage |
| :--- | :--- | :--- |
| [`RELATED_WORK_LITERATURE_SURVEY.md`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_WORK_LITERATURE_SURVEY.md) | Comprehensive 6-pillar literature survey analyzing the full scientific landscape, theoretical underpinnings, and research gaps. | Primary reference for understanding the broader academic context, writing long-form thesis chapters, or preparing defense slides. |
| [`COMPARATIVE_TAXONOMY_MATRIX.md`](file:///E:/Akhi%20Thesis/Related%20paper/COMPARATIVE_TAXONOMY_MATRIX.md) | Structured comparative tables and feature matrices comparing CMAT-HOI against 12+ baselines across 10 technical dimensions. | Perfect for copying comparative tables into Section 2 of your conference paper or thesis. |
| [`PAPER_SUMMARIES_AND_CONTRASTS.md`](file:///E:/Akhi%20Thesis/Related%20paper/PAPER_SUMMARIES_AND_CONTRASTS.md) | Detailed 1-page structured breakdowns of the top 15 most critical related papers, highlighting why each fails and how CMAT-HOI overcomes it. | Use when addressing specific peer-review critiques, authoring rebuttal letters, or citing specific models. |
| [`RELATED_WORK_SECTION_DRAFT.tex`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_WORK_SECTION_DRAFT.tex) | Ready-to-publish, clean LaTeX `\section{Related Work}` divided into 4 cohesive, scholarly subsections with active academic voice. | Direct drop-in replacement for Section 2 in your Overleaf IEEE/ACM conference LaTeX project. |
| [`RELATED_PAPERS_BIBLIOGRAPHY.bib`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_PAPERS_BIBLIOGRAPHY.bib) | Verified, clean BibTeX database with complete citation metadata (DOIs, page ranges, volumes, publishers) for all 30+ surveyed papers. | Import directly into your Overleaf or LaTeX `.bib` file (`references.bib`). |

---

## 🔬 The 4 Core Pillars of Your Related Work Section

Your research paper's related work should be organized around these four thematic pillars:

```
                               ┌──────────────────────────────────────────────────────────┐
                               │           CMAT-HOI RELATED WORK ARCHITECTURE             │
                               └────────────────────────────┬─────────────────────────────┘
                                                            │
                 ┌──────────────────────────┬───────────────┴──────────────┬──────────────────────────┐
                 │                          │                              │                          │
                 ▼                          ▼                              ▼                          ▼
     [ Pillar 1: Sports Video ]    [ Pillar 2: Skeleton GCNs ]    [ Pillar 3: HOI & V-HOI ]  [ Pillar 4: Multimodal ]
     • 3D CNNs (I3D, SlowFast)     • ST-GCN (Yan et al.)          • Static HOI (QPIC, HOTR)   • Representation Collapse
     • ViTs (ViViT, Video Swin)    • Adaptive GCNs (2s-AGCN)      • Video HOI (VidHOI)        • Modality Stems (d=256)
     • FineGym (Shao et al.)       • PoseConv3D (Duan et al.)     • Action Genome (Ji et al.) • Dynamic Gating (GMFU)
     • Scene Bias Limitation       • Object Blindness Gap         • Cross-Attention Binding   • Multi-Scale 4-Way Head
```

### 1. Video Action Recognition in Sports Analytics
- **Foundational Works**: Two-Stream ConvNets (Simonyan & Zisserman, 2014), I3D (Carreira & Zisserman, CVPR 2017), SlowFast (Feichtenhofer et al., ICCV 2019), ViViT (Arnab et al., ICCV 2021), Video Swin Transformer (Liu et al., CVPR 2022).
- **Sports Benchmarks**: UCF-101 (Soomro et al., 2012), FineGym (Shao et al., CVPR 2020), Sports Action Surveys (Host et al., IEEE Access 2022).
- **Critical Failure Mode**: **Scene Bias** (exploiting static stadium/court backgrounds rather than the athlete's actual motor execution).

### 2. Skeleton-Based Action Recognition and Its Limitations
- **Key Methodologies**: ST-GCN (Yan et al., AAAI 2018), 2s-AGCN (Shi et al., CVPR 2019), CTR-GCN (Chen et al., ICCV 2021), PoseConv3D (Duan et al., CVPR 2022), ST-TR (Plizzari et al., 2021).
- **Critical Failure Mode**: **Object Blindness** (pure anatomical joint graphs cannot distinguish actions using identical arm swings with different sports gear, e.g., cricket bat vs. baseball bat vs. tennis racket).

### 3. Human–Object Interaction (HOI) in Static and Dynamic Contexts
- **Static Formulations**: InteractNet (Gkioxari et al., CVPR 2018), HICO-DET (Chao et al., WACV 2018), QPIC (Tamura et al., CVPR 2021), HOTR (Kim et al., CVPR 2021).
- **Video HOI Extensions**: Action Genome (Ji et al., CVPR 2020), VidHOI (Chen & Chao et al., 2021).
- **Critical Failure Mode**: Static HOI ignores temporal momentum; video scene graphs suffer from combinatorial computational explosion.
- **CMAT-HOI Solution**: Lightweight Bi-Directional Cross-Attention ($h=4$) binding joint keypoints directly to sports apparatus boxes.

### 4. Multimodal Transformers and Gradient Balancing
- **Optimization Challenges**: "What Makes Training Multi-modal Networks Hard?" (Wang, Tran, Feiszli, NeurIPS 2020), Modality Imbalance (Wu et al., NeurIPS 2022).
- **Multimodal Architectures**: MulT (Tsai et al., ACL 2019), MBT Attention Bottlenecks (Nagrani et al., NeurIPS 2021), Gated Multimodal Units (Arevalo et al., 2017).
- **Critical Failure Mode**: **Multimodal Representation Collapse** (high-dimensional visual features dominate $>95\%$ of dimensions and $>93\%$ of backprop gradients, suppressing kinematic cues).
- **CMAT-HOI Solution**: Dimension-Aligned Stems ($d=256$) with 1D temporal convolution equalizing gradient flow, coupled with GMFU softmax gating.

---

## 🚀 How to Integrate This into Your Paper

1. **For Overleaf / LaTeX Conference Paper (`overleaf_conference_paper.tex` / `main.tex`)**:
   - Replace your existing `\section{Related Work}` with the ready-to-publish text from [`RELATED_WORK_SECTION_DRAFT.tex`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_WORK_SECTION_DRAFT.tex).
   - Append the entries from [`RELATED_PAPERS_BIBLIOGRAPHY.bib`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_PAPERS_BIBLIOGRAPHY.bib) into your project's `bibliography.bib` or `references.bib`.

2. **For Full Master's Thesis Chapter 2 (Literature Review)**:
   - Use [`RELATED_WORK_LITERATURE_SURVEY.md`](file:///E:/Akhi%20Thesis/Related%20paper/RELATED_WORK_LITERATURE_SURVEY.md) as the narrative blueprint, expanding each subsection with the mathematical equations and figures provided.
   - Insert the comparative tables from [`COMPARATIVE_TAXONOMY_MATRIX.md`](file:///E:/Akhi%20Thesis/Related%20paper/COMPARATIVE_TAXONOMY_MATRIX.md) into Section 2.5 of your thesis to summarize the literature gaps.
