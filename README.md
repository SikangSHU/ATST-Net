# ATST-Net

## Advancing Stain Transfer for Multi-Biomarkers: A Human Annotation-Free Method Based on Auxiliary Task Supervision (IJCAI 2025)

---

## Instructions

### 1. Dataset Preparation

Crop all tiles in the original dataset into **512×512** patches.

Run `dataset_preprocessing/IHC_autoanno_otsu_threshold.py` to generate stain-unmixed images and masks. Organize the dataset of each biomarker following the examples in the `dataset` folder.

Ground-truth positive nucleus counts (`positive_nucleus_num_gt_512.npz`) are obtained by:

- `nucleus_gt_generate.py`, and
- the trained nucleus segmentation network **CPP-Net** [1].

---

### 2. Pretrained Models

Download the original pretrained model **PathoDuet** [2]:

> https://github.com/openmedlab/PathoDuet

Place the original PathoDuet pretrained checkpoint and the trained CPP-Net checkpoint in:

```
/pretrained_pathoduet/pretrained_model
/cppnet_checkpoint/IHC_nucleus_stage2
```

---

### 3. Environment Setup

```bash
conda env create -f environment.yml
```

---

### 4. Training

```bash
python ATSTNet_main_train.py
```

---

### 5. Testing

```bash
python ATSTNet_main_test.py
```

---

## References

[1] Chen S, Ding C, Liu M, et al.  
**CPP-net: Context-aware polygon proposal network for nucleus segmentation.**  
_IEEE Transactions on Image Processing,_ 2023, 32: 980-994.

[2] Hua S, Yan F, Shen T, et al.  
**PathoDuet: Foundation models for pathological slide analysis of H&E and IHC stains.**  
_Medical Image Analysis,_ 2024, 97: 103289.
