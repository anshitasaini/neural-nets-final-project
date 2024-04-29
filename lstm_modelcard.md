---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for Jeopardy LSTM Model

A bi-directional LSTM trained on a dataset of Jeopardy questions and their values to predict the value of a given Jeopardy question.

## Model Details

### Model Description

A bi-directional LSTM trained on a dataset of Jeopardy questions and their values to predict the value of a given Jeopardy question

- **Developed by:** Anshita Saini, Nihita Sarma, Nishitha Vattikonda
- **Model type:** Bi-Directional LSTM
- **Language(s) (NLP):** English
- **License:** Other

### Model Sources

- **Repository:** https://github.com/anshitasaini/neural-nets-final-project

## Uses

### Direct Use
This model is made to predict a value category when provided with a Jeapordy Question.

### Downstream Use
This model could possibly be extended to predict values for questions/statements in other datasets. If altered and fine-tuned, it could also possibly be trained to generate Jeopardy questions. It could also be extended to predict the exact value of the Jeopardy question rather than a category or bucket of values.

### Out-of-Scope Use
This model cannot be used for tasks that are not in the format of predicting a value from a prompt or question. It also should not be used in applications where high accuracy is necessary.

## Bias, Risks, and Limitations
This model was developed to satisfy curiosity and should not be used for any other purpose due to it's accuracy and bias towards predicting specific values. It only works on padded Jeopardy Questions and has an accuracy bias towards certain question categories, making it extremely limited.

### Recommendations
This model was developed to satisfy curiosity and should not be used for any other purpose due to it's accuracy and bias towards predicting specific values. It is recommended that this model is only analyzed and used as an educational tool to explore LSTMs due to its limitations.

## How to Get Started with the Model

Use the code below to get started with the model.

https://github.com/anshitasaini/neural-nets-final-project

## Training Details

### Training Data
200,000+ Jeopardy Questions (Kaggle): https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions/code \n\nThis is a dataset containing 200,000 Jeopardy questions, their answers, categories, values, rounds, and air dates.

### Training Procedure

#### Preprocessing
First, the questions were extracted from the data, questions with values over 2000 were removed, and the remaining questions were converted to integer sequences using a char to int mapping. Then, they were all padded to the same length and the values were classified into buckets. Lastly, these new question value pairs were divided into a train, validation, and test set.

#### Training Hyperparameters

- **Training regime:** The model had a hidden size of 256. It was trained with an Adam optimizer and CrossEntropyLoss function for 100 epochs. 

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
The testing data was taken from the same source as the training data and had the same preprocessing with a train test split.

#### Factors
Question category, value, and question length can all influence model behavior.

#### Metrics
This model was tested based on overall accuracy and accuracy by category.

### Results
This model has a 23% testing accuracy.

## Model Examination
To interpret model behavior, both categorical accuracies and average values per category were analyzed. The model's performance was also compared to that of an LR model trained on the same data

## Technical Specifications

### Model Architecture and Objective
This model is a bidirectional LSTM that takes in a question, gets a positional encoding from it, puts that through an LSTM layer, and puts that through a linear layer to get a value output.

### Compute Infrastructure
No computer infastructure beyond an average laptop is needed for this model to be trained or run.

#### Hardware
Only a CPU and an average laptop are required to run and train this model.

#### Software
Pytorch, numpy, and pandas are required to train and run this model.

## Model Card Contact
Contact nihitasarma@utexas.edu for more information.
