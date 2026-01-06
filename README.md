<p align="center">
  <h1 align="center">GEOREASON: ALIGNING THINKING AND ANSWERING IN REMOTE SENSING VISION-LANGUAGE MODELS VIA LOGICAL CONSISTENCY REINFORCEMENT LEARNING</h1>
  <p align="center">
      <a href='https://scholar.google.com/citations?hl=en&user=TvsTun4AAAAJ' style='text-decoration: none' >Wenshuai Li</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=A39S7JgAAAAJ&hl=en' style='text-decoration: none' >Xiantai Xiang</a><sup></sup>&emsp;
      <div align="center">
      <a href='https://arxiv.org/abs/2501.09720'><img src='https://img.shields.io/badge/arXiv-2501.09721-brown.svg?logo=arxiv&logoColor=white'></a>
      <a href='https://huggingface.co/collections/Qingyun/lmmrotate-6780cabaf49c4e705023b8df'><img src='https://img.shields.io/badge/HuggingFace-Model-yellow.svg?logo=HuggingFace&logoColor=white'></a>
      </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

---

## GeoReason: Overview
The advancement of Remote Sensing Vision-Language Models (RS-VLMs) necessitates a paradigm shift from recognition-centric perception to high-level deductive reasoning. However, current models often suffer from logical hallucinations, where correct answers are derived from flawed reasoning chains or rely on positional shortcuts rather than spatial logic. To address these challenges, we introduce GeoReason-Bench, a logic-driven dataset curated through an expert-knowledge pipeline that transforms raw geometric primitives and morphological patterns into high-fidelity reasoning trajectories. We further propose a two-stage training pipeline to internalize these cognitive capabilities. In the first stage, Supervised Knowledge Initialization aligns the model with Chain-of-Thought (CoT) syntax and domain expertise. In the second stage, Consistency-Aware Reinforcement Learning employs Group Relative Policy Optimization (GRPO) to refine deductive reliability. Central to our approach is a novel Logical Consistency Reward (LCR) mechanism, which utilizes an Option Permutation strategy to penalize logical drift and ensure that model decisions are strictly anchored in verifiable reasoning traces. Experimental results demonstrate that our framework significantly enhances the cognitive reliability and interpretability of RS-VLMs, enabling expert-level decision-making in complex remote sensing scenarios.
<p align="center"><img src="assets/pipeline.png" width="80%"></p>

---

## Performance
Eevaluating models across Perceptual Tasks (Count, Color, Shape, Scene) and Reasoning Tasks (Reason) to analyze their multi-level understanding.
<p align="center"><img src="assets/result.png" width="80%"></p>

---

## Get Started

### Environment Installation

```bash
conda create -n GeoReason python=3.10
conda activate GeoReason
pip install -r requirements.txt
```
