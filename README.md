# CytoSet

---

## Introduction

CytoSet is a deep-learning based method used for predicting clinical outcome from cytometry data.

<p align="center">
<img align="middle" src="./assets/CytoSet.png" alt="CytoSet" width="600" />
</p>

## Installation

### Requirements

- Python >= 3.6
- CUDA >= 10.1

```
pip install -r requirements.txt
```

### Datasets

AML dataset is from https://flowrepository.org/id/FR-FCM-ZZYA. HEUvsUE dataset is from https://flowrepository.org/id/FR-FCM-ZZZU.
ICS dataset is from https://flowrepository.org/id/FR-FCM-ZZZV. NK_cell dataset is from the repository of [CellCNN](https://github.com/eiriniar/CellCnn). 

The pre-processed dataset for training the model can be downloaded from the [google drive](https://drive.google.com/drive/folders/1chfsOYSCsRg7kcydqyrze6B1Sg9-iInL?usp=sharing).


## Reproducing Results

### Training

* Download pre-processed the datasets (see Datasets Section) and unpack them.
* In ``scripts/train/train_[Dataset].sh``, set ``bin_file`` to the path of ``train.py`` and ``gpu`` to the gpu id.
* Start training: ``bash train_[Dataset].sh``

### Testing

* We provide our pre-trained model on HEUvsUE dataset and test dataset in ``checkpoints``.
* We also provide our model configuration for each dataset in ``config/model``.
* To run the testing, you can use the following command:

```
python test.py --model checkpoints/HEUvsUE_model.pt --config config/model/HEUvsUE/config.json --test_pkl checkpoints/test_HEUvsUE.pkl
```

The evaluation results are:

| Accuracy  | Area Under Curve |
|-----------|------------------|
|   0.903   |     0.931        |


## Citing

TBD


## Contact

If you have any questions, please feel free to contact Haidong Yi (haidyi@cs.unc.edu) or push an issue on Issues Dashboard.


