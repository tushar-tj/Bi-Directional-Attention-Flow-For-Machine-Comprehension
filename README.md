# BiDaF-Squad-1.0
The repo contains an implementation of Bidirectional Attention Flow for Machine Comprehension on Squad v1.0 & Cloze-Style Reading Comprehension as illustrated in the paper from Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi

## Training Results

* Training Results

|                  **DATASET**                  |                 **RESULTS**           |
|---------------------------------------------- |---------------------------------------|
|                 SQuAD v1.1                    |       |
| Cloze-Style Reading Comprehension (CNN)       |                                  |
| Cloze-Style Reading Comprehension (DailyMail) |                                  |

* Training Results Original Paper

|                  **DATASET**                  |                 **RESULTS**           |
|---------------------------------------------- |---------------------------------------|
|                 SQuAD v1.1                    |  Exact Match - 67.7%,  F1 - 77.3%     |
| Cloze-Style Reading Comprehension (CNN)       |               76.3%                   |
| Cloze-Style Reading Comprehension (DailyMail) |               80.3%                   |

The Results are without hyperparameter tuning models, tuning could be applied to achieve results much closer to original paper.


## Getting Started

Dataset preprocessing is required to flatten the dataset.  
When first running the train process use -load_data=True

* Training **BIDAF** model on **SQUAD v1.1**
```
python train.py -model_name=<model_name> -dataset_name=SQUAD
```

* Training **BIDAF** model on **Cloze-Style Reading Comprehension**
```

```

#### EXAMPLE

```
python train.py -model_name=LOCAL -dataset_name=SQUAD
```

*Note: Please check parameters section for complete details.*
*Also, saving the model would slow down the training process.*

### Prediction
To extract response from a context

```
python predict.py -context=<context> -query=<query> -model_path=<path of trained model> -word_vocab=<path of WORD vocab> -char_vocab=<path of CHAR vocab>
```
####Example
```
python predict.py -context="He was speaking after figures showed that the country's economy shrank by 20.4% in April - the largest monthly contraction on record - as the country spent its first full month in lockdown." -query="By how much did the country's economy shrank" -model_path=./model_checkpoints/best_local_squadv1.1.torch -word_vocab=./model_checkpoints/SQUAD_WORDS.vocab -char_vocab=./model_checkpoints/SQUAD_CHAR.vocab

Time: 0:00:01 , ANSWER: 20 . 4 %
```

## Prerequisites

```
torch==1.4.0
torchtext==0.6.0
pandas==1.0.3
```

## Parameters

```bash
python train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  -data DATA            Path to input data
  -data_checkpoint DATA_CHECKPOINT
                        Path to store preprocessed data checkpoints
  -model_checkpoint MODEL_CHECKPOINT
                        Path to store modelled data checkpoints
  -model_name MODEL_NAME
                        provide a name to the model for storing at chekpoints
  -load_data [LOAD_DATA]
                        To Load data of use preprocessed data
  -dataset_name DATASET_NAME
                        Name of the Dataset
  -epochs EPOCHS        No. of Epoch to run
  -batch_size BATCH_SIZE
                        Number of examples in each batch
  -glove_size GLOVE_SIZE
                        Size of Glove vector to use
  -char_embedding_size CHAR_EMBEDDING_SIZE
                        Size of Character embeddings to be used
  -kernel_size KERNEL_SIZE
                        Kernel Size
  -channels_count CHANNELS_COUNT
                        Count of channels for character embeddings
  -learning_rate LEARNING_RATE
                        Learning Rate
  -epoch_log EPOCH_LOG  Print logs after xx epochs
```

```bash
python predict.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  -context CONTEXT      Context in text
  -query QUERY          Query in text
  -model_path MODEL_PATH
                        path to model
  -word_vocab WORD_VOCAB
                        path to word vocab
  -char_vocab CHAR_VOCAB
                        path to char vocab

```