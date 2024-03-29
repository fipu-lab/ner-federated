# Federated Learning of BERT models in natural language processing

Federated Learning using pretrained [BERT](https://github.com/google-research/bert) models for a Named Entity Recognizition (NER) task. 

Experiments were run on [Few-NERD](https://github.com/thunlp/Few-NERD) and [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) datasets.


## Run code

To run the code, first you need to download BERT config from [google-research](https://github.com/google-research/bert). Since the original models were trained using Tensorflow v1.x, checkpoints need to be converted to TF v2. Use [this script](https://colab.research.google.com/gist/infinex/f0e243e6da66124fb1e0fd4d9f830909/tf1-x-to-tf2-weight-transfer-for-bert.ipynb#scrollTo=yjaAb_NLxUlg) to convert bert checkpoints from TF1 to TF2. Unzip desired BERT model zip into `BERT-NER-TF2/models` folder and use the model by its name in notebooks:

- [FL.ipynb](FL.ipynb) - training of pretrained or randomly initialized models
- [FL_frozen-bert.ipynb](FL_frozen-bert.ipynb) - fine-tuning of pretrained models by freezing the BERT model layers
- [Central](Central.ipynb) - training of one centralised model

Datasets are already included in this repository, and are located in [BERT-NER-TF2/dataset](BERT-NER-TF2/dataset/) directory.


## About

* This source code is a result of a Fipulab project:
  * Juraj Dobrila University of Pula
  * Faculty of informatics in Pula
* Contributors: Mateo Borina, Robert Šajina, Nikola Tanković
* Topic / title: Federated Learning of BERT models in natural language processing
* Project status: working

## Features

* Federated learning
* BERT models (transformer-based neural networks)
* Natural language processing
* Named entity recognition tasks

## Code

* Python
* Jupyter Notebook
* Results are being saved to JSON
* Digrams in PNG format
* Datasets: CoNLL-2003, Few-NERD

## Other

* FIPU: https://fipu.unipu.hr/
* UNIPU: https://www.unipu.hr/

