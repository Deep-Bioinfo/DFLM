# DFLM: Transcription Factors Fine-tune Language Model for mining transcription factor binding sites


## 1. Data Preparation and Environment setup

DFLM mainly used two types of data sets, the unlabeled large-scale
[Genome Reference Consortium Human Build
38](https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.26/)(GRCh38/hg38)
and about 500 labeled [ChIP-seq data
set](http://tools.genes.toronto.edu/deepbind/nbtcode/). 

We recommend you to build a python virtual environment with Anaconda. Also, please make sure you have at least one NVIDIA GPU with Linux x86_64 Driver Version <= 384.145 (compatible with CUDA 9.0). 

#### 1.1 Create and activate a new virtual environment

```
conda create -n DFLM python=3.7
conda activate DFLM
```

#### 1.2 Install the package and other requirements


```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/Deep-Bioinfo/DFLM
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt
```
## 2. DFLM Overview

Now that we can process genomic data into a form we can feed to a model,
we need to determine our strategy for training the model. Lets start by
defining our end goal: We want to train a sequence model to classify
genomic sequences using sequence input alone. This poses a potential
problem. Sequence models tend to require a large amount of data to train
effectively, and labeled genomic classification datasets can be small.
The DFLM approach provides a solution to this. DFLM breaks training
into three stages:

1.  First we train a general domain language model using unsupervised on
    a large unlabeled corpus hg38.
2.  We fine tune the general language model on the ChIP-seq data
    sets to create a task specific language model
3.  We fine tune the task specific language model for classification

#### Language Model 

A language model is trained in an unsupervised fashion, meaning that no
labeled data is required. Since the goal of the language model is to
predict the next k-mer in a sequence, each k-mer becomes a correct
output prediction for the sequence that preceeds it. This means we can
generate huge amounts of paired data (input sequence + next k-mer) from
any unlabeled genomic sequence.

#### Classification Model

A classification model is trained in a supervised fashion, requiring
paired labeled data. For example if the task is promoter classification,
all sequences in the classification dataset must be labeled as 0 or 1
for not-promoter or promoter.

When we transfer to the classification model, we only transfer the
Embedding and the Encoder, as the classifcation model required a
different linear head. Visually:

![avatar](https://github.com/Deep-Bioinfo/DFLM/blob/main/DFLM.png)
<center> Fig.1 The architecture of DFLM (Blue arrows show transfer learning) </center>

The arthitectures for the Classification Model and the Language Model
follow similar structures - the consist of an **Embedding**, an
**Encoder**, and a **Decision**. On a high level, these layers function
in the following ways:

-   Embedding: Converts the numericalized tokens of the input sequence
    into vector representations
-   Encoder: Processes the vectorized sequence into a hidden state by
    ASGD Weight-Dropped LSTM (AWD-LSTM)
-   Decision: Uses the hidden state to make a classification decision.

#### ASGD Weight-Dropped LSTM (AWD-LSTM)

AWD-LSTM is a variant of LSTM. It is one of the best language modeling
models in NLP. Encoder layer is composed of 6 layers of ASGD
Weight-Dropped LSTM (AWD-LSTM). We know that dropout has been a great
success in mitigating over-fitting. However, applying dropout techniques
on RNNs is not effective because dropout zeroes in on a randomly
selected subset of activations in each layer, which interferes with the
RNN\'s ability to maintain long-time dependencies. Therefore, instead of
activations, AWD-LSTM zeroes out a randomly chosen subset of weights,
with each unit receiving input from a random subset of units in the
previous layer. It has also been found that the traditional SGD
algorithm without driving weights is better than other algorithms such
as momentum SGD, Adam, Adagrad, and RMSProp in language modeling tasks.
Therefore, AWD--LSMT uses a variant of the traditional SGD algorithm,
ASGD (Average SGD). It uses the same gradient update step as the SGD
algorithm, but instead of returning the weights computed in the current
iteration, it also considers the weights of the previous iterations and
returns a mean value. These strategies are not only very efficient, but
also remain the same as the existing LSTM structure.
