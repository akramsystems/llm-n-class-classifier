# LLM Classification API

## Overview

This project is a FastAPI application designed to perform classification tasks using a language model (LLM). It supports custom labels, few-shot examples, and testing with multiple datasets.

## Mathematical Problem Formulation
    
### Given:

Set of inputs `X={x1,x2,…,xn}:`
The input data points that need classification.

Set of user-defined labels `L={l1,l2,…,lk}:`
Custom labels provided by end-users, along with their descriptions
`D={d1,d2,…,dk}.`

Classification tasks:
    Binary classification `(L={0,1})`
    Multi-class classification `(|L|>2)`

Few-shot examples `F={(xf,lf)}f=1m`:
A small set of labeled examples provided for the system to learn from,
where `xf∈X` and `lf∈L`.

### Objective:

Design a system S such that:

`S: X → L`, where `S(x)` predicts a label `l` in `L` for each input `x` in `X`.

## Usage

To classify input data, send a POST request to the `/classify` endpoint with the dataset name, input data, optional few-shot examples, and custom labels.

### Endpoint

- **POST /classify**: Classifies the input data using the LLM.
  - **Parameters**:
    - `dataset_name` (str): Name of the dataset, e.g., 'iris', 'mnist', 'imdb'.
    - `input_data` (str): Raw input to classify. Could be text or a reference to features.
    - `custom_labels` (List[CustomLabel]): List of custom labels of possible output values and their descriptions.
    - `few_shot_examples` (Optional[List[Dict]]): Optional few-shot examples to be appended to the system prompt.
    - `input_schema_definition` (Optional[str]): Optional input schema definition to be appended to the system prompt.

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Running the Application

1. **Build the Docker Image**:
   ```bash
   docker-compose build
   ```

2. **Start the Docker Container**:
   ```bash
   docker-compose up
   ```

3. **Access the API**:
   The API will be available at `http://localhost:8000`.

### Example Request and Response

To classify input data, send a POST request to the `/classify` endpoint. Below is an example of how to structure your request and the expected response.

#### Request Example
![Example Request](docs/example-request.png)

#### Response Example
![Example Response](docs/example-response.png)

### Executing a Python Function within the Docker Container

To execute a Python function within the Docker container, follow these steps:

1. **Access the Running Container**:
   ```bash
   docker exec -it <container_id> /bin/bash
   ```

   Replace `<container_id>` with the actual container ID, which you can find using `docker ps`.

2. **Run a Python Script**:
   Once inside the container, you can execute any Python script. For example:
   ```bash
   python src/scripts/download_datasets.py
   ```


## Running Tests and Scripts with Docker Compose

The following datasets were used for testing:

- **Iris**: Tabular data with custom labels like "setosa", "versicolor", "virginica" for multi-class classification.

- **IMDB**: Text data with sentiment labels for binary classification ["positive", "negative"].

- **NSL-KDD**: Cyber-security dataset with labels ["anomaly", "normal"] for binary classification.

### Make sure image is running and get container id

```bash
$ docker-compose up
$ docker ps
```

for example, container id is `308152ef22c2`

### Download the Datasets First (if not already downloaded)

```bash
$ docker exec -it 308152ef22c2 python -m scripts.download_datasets
```

### Running Tests

```bash
$ docker exec -it 308152ef22c2 python -m tests.test_classification
```

## Additional Notes

- Problem Formulation:
    - The case of having a binary classification problem is a special case of the general classification problem.
    - This assumes that we have aleast 2 classes even in the binary case this way it we can generalize the problem to a multi-class classification problem where N >= 2.

- **Model Choice**: 

    - We could have used a SLM which does dynamic classification, that is trained.  This could be useful if we wanted to save money on the LLM calls. but the caveat would not be able to add context to our input which is needed to understand the variables if some of the variables are not text based and numbers. 
    - An example of this would be available here on hugging face i.e. [Facebooks BART-Large-MNLI](https://huggingface.co/facebook/bart-large-mnli)

- **Dataset**: 
    - We could have used a larger sample size for our testing but i did this to keep it simple and shuffled the data and randomly sampled 100 rows. Now if our possible custom labels are a large number we would need to sample more data relative to the number of labels we have.

- **Testing**: 
    - I didn't include tests with the few shot examples because I wanted to get this PR in, now wew could have sampled some of the data from the dataset and used them as few shot examples if we had an example dataset, specifically sampling atleast M examples for each label.
    - We use simply accuracy where as we could have used other metrics like F1 score, precision, recall, etc. and output a confusion matrix to accompany the tests.
    - We are likely to hit rate limits on the LLM calls so we would need to implement a rate limiting mechanism.
    - Didn't create unit tests but that is a good idea.
    - We can encorporate an MLOPs tool like [ClearML](https://clear.ml/) to track the experiments and the results on CI/CD, this way we can have a history of the experiments and the results.

- **Future Optimization**: 
    
    - _Better Schema_: We aren't dynamically generating the Basemodel Resposne for each dataset we could choose to dynamically generate a BaseModel based on the `custom_labels` which are passed in this way we can have a stronger type check enforced by the llms `.parse()` method as it is possible for it to NOT return an expected output. 
    
    - _Better Feature Engineering_: We could do a type of feature optimization where we find the most important features for the classification task using a method like PCA this way we can ensure the number of features we us to handle the classification task is minimized.

    - _Better Model or Multi Model Approach_: We can compare and contrast this approach to that of using embeddings and a transformer model like BERT or GPT-3.5-turbo. Or try to take a look at ReRanker Models. To see if that is a better way to fit the problem criteria.  Multiple Models could also be used and we can use a voting mechanism to determine the final output.

