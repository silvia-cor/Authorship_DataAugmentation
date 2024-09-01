# Data Augmentation for Authorship Verification

## Abstract

**Authorship Verification (AV)** is a text classification task, where the goal is to infer whether a text has been written by one specific author or by someone else.

It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. 

In this project, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest.

## The datasets

In this project, we employ 5 different datasets:
- *TweepFake*: a limited version is available on Kaggle [here](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text);
- *EBG*: it can be downloaded [here](https://zenodo.org/record/5213898##.YuuaNdJBzys), in the -dataset argument you can specify whether you want obfuscated texts in the tests set (ebg-obf) or not (ebg);
- *RJ*: it can be downloaded [here](https://zenodo.org/record/5213898##.YuuaNdJBzys); in the -dataset argument you can specify whether you want obfuscated texts in the tests set (rj-obf) or not (rj);
- *PAN11*: it can be downloaded [here](https://pan.webis.de/clef11/pan11-web/authorship-attribution.html);
- *Victoria*: it can be downloaded [here](https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution).

For each dataset, we divide the texts into chunks. 

Each AV experiment is done by iteratively taking each author in the dataset as the author of interest.


## The models and training policies

We experiment with three different generator architectures:
- GRU 
- Transformer
- Gpt2

with two training policies for the generators:
- one inspired by standard Language Models (LMT)
- one inspired by Wasserstein Generative Adversarial Networks (GANT)

and with two learning algorithms for the AV classifier:
- Support Vector Machine (SVM) 
- Convolutional Neural Network (NN). 


### Code 

The code is organized as follows in the `src` directory:
- `main.py`
- `Training_Generation.py`: contains the pipeline for the training of the generators, and the actual generation process.
- `BaseFeatures_extractor.py`: contains the pipeline to extract the BaseFeatures.
- `generators`: directory with the code for the generator models (gpt2: `generator_gpt2.py`, gru/transformer outputting embeddings: `generator_embed.py`, gru/transformer outputting onehot vectors: `generator_onehot.py`).
- `classifiers`: directory with the code for the classifier models (SVM: `classifier_svm.py`, NN: `classifier_nn.py`); each one also contains the pipeline for the classification.
- `general`: directory with the code for various purposes (plotting: `visualization.py`, test for statistical significance: `significance_test.py`, processing the datasets: `dataloader.py` and `process_dataset.py`).


### References

(preliminary experiments)

Silvia Corbara, Alejandro Moreo. 2023. Enhancing Adversarial Authorship Verification with Data Augmentation. 13th Italian Information Retrieval Workshop (IIR 2023). https://ceur-ws.org/Vol-3448/paper-11.pdf


# Quickstart
## Installation
```shell
mkvirtualenv -p python3.9 av_aug
source av_aug/bin/activate
pip install -r requirements.txt
```
Remember to download the dataset(s), and to update the dataset folder in the variable DATA_PATH in `main.py` if needed.
## Run
Example to run the AV experiment on the pan11 dataset, using the SVM classifier and no augmentation.
```
python main.py -dataset pan11 -classifier svm
```
When using the NN classifier, you want to also specify the generator, even if you don't employ the data augmentation. This is because there is a different NN architecture for each kind of generator output.

Example to run the AV experiment on the victoria dataset, using the NN classifier (with the architecture to process inputs in the shape of GPT2 outputs) and no augmentation.
```
python main.py -dataset victoria -classifier nn --generator gpt2
```
Example to run the AV experiment on the TweepFake dataset, employing data augmentation and using:
- the NN classifier with appropriate architecture
- the Transformer 
  - outputting embeddings (embedTrans)
  - trained as Language Model (LMT)
- the GPUs labelled 1 and 2.
```
python main.py -dataset tweepfake -classifier nn --generator embedTrans --training_policy LMT --devices 1,2
```
## Output
The processed dataset is stored as a pickle file for future use (see PICKLES_PATH in `main.py`). Similarly, the various models (both generators and classifiers) are stored in the respective directory (see MODELS_PATH in `main.py`).

The results of the classification are stored as:
- a pickle file with the actual predictions (see PICKLES_PATH in `main.py`)
- a csv file with the computed metrics (see RESULTS_PATH in `main.py`).