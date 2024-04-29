---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for {{ model_id | default("Model ID", true) }}

{{ model_summary | default("A bi-directional LSTM trained on a dataset of Jeopardy questions and their values to predict the value of a given Jeopardy question.", true) }}

## Model Details

### Model Description

{{ model_description | default("A bi-directional LSTM trained on a dataset of Jeopardy questions and their values to predict the value of a given Jeopardy question.", true) }}

- **Developed by:** {{ developers | default("Ashita Saini, Nihita Sarma, Nishitha Vattikonda", true)}}
- **Model type:** {{ model_type | default("Bi-Directional LSTM", true)}}
- **Language(s) (NLP):** {{ language | default("English", true)}}
- **License:** {{ license | default("other", true)}}

### Model Sources

- **Repository:** {{ repo | default("https://github.com/anshitasaini/neural-nets-final-project", true)}}

## Uses

### Direct Use

{{ direct_use | default("This model is made to predict a value category when provided with a Jeapordy Question.", true)}}

### Downstream Use

{{ downstream_use | default("This model could possibly be extended to predict values for questions/statements in other datasets. If altered and fine-tuned, it could also possibly be trained to generate Jeopardy questions. It could also be extended to predict the exact value of the Jeopardy question rather than a category or bucket of values.", true)}}

### Out-of-Scope Use

{{ out_of_scope_use | default("This model cannot be used for tasks that are not in the format of predicting a value from a prompt or question. It also should not be used in applications where high accuracy is necessary.", true)}}

## Bias, Risks, and Limitations

{{ bias_risks_limitations | default("This model was developed to satisfy curiosity and should not be used for any other purpose due to it's accuracy and bias towards predicting specific values. It only works on padded Jeopardy Questions and has an accuracy bias towards certain question categories, making it extremely limited.", true)}}

### Recommendations

{{ bias_recommendations | default("This model was developed to satisfy curiosity and should not be used for any other purpose due to it's accuracy and bias towards predicting specific values. It is recommended that this model is only analyzed and used as an educational tool to explore LSTMs due to its limitations.", true)}}

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("https://github.com/anshitasaini/neural-nets-final-project", true)}}

## Training Details

### Training Data

{{ training_data | default("200,000+ Jeopardy Questions (Kaggle): https://www.kaggle.com/datasets/tunguz/200000-jeopardy-questions/code \n\nThis is a dataset containing 200,000 Jeopardy questions, their answers, categories, values, rounds, and air dates.", true)}}

### Training Procedure

#### Preprocessing

{{ preprocessing | default("First, the questions were extracted from the data, questions with values over 2000 were removed, and the remaining questions were converted to integer sequences using a char to int mapping. Then, they were all padded to the same length and the values were classified into buckets. Lastly, these new question value pairs were divided into a train, validation, and test set.", true)}}

#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("The model had a hidden size of 256. It was trained with an Adam optimizer and CrossEntropyLoss function for 100 epochs.", true)}} 

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

{{ testing_data | default("The testing data was taken from the same source as the training data and had the same preprocessing with a train test split.", true)}}

#### Factors

{{ testing_factors | default("Question category, value, and question length can all influence model behavior.", true)}}

#### Metrics

{{ testing_metrics | default("This model was tested based on overall accuracy and accuracy by category.", true)}}

### Results

{{ results | default("This model has a 23% testing accuracy.", true)}}

## Model Examination

{{ model_examination | default("To interpret model behavior, both categorical accuracies and average values per category were analyzed. The model's performance was also compared to that of an LR model trained on the same data", true)}}

## Technical Specifications

### Model Architecture and Objective

{{ model_specs | default("This model is a bidirectional LSTM that takes in a question, gets a positional encoding from it, puts that through an LSTM layer, and puts that through a linear layer to get a value output.", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("No computer infastructure beyond an average laptop is needed for this model to be trained or run.", true)}}

#### Hardware

{{ hardware_requirements | default("Only a CPU and an average laptop are required to run and train this model.", true)}}

#### Software

{{ software | default("Pytorch, numpy, and pandas are required to train and run this model.", true)}}

## Model Card Contact

{{ model_card_contact | default("Contact nihitasarma@utexas.edu for more information.", true)}}
