
# RNN + Attention Named Entity Recognition  

Named Entity Recognition model using BiLSTM with Attention network.
For feature embedding uses glove embeddings. 
CRF is used on the top of Attention layer for predicting the Entity Tags. 



## Model Architecture

![Architecture](https://github.com/upjabir/lstm_ner/blob/main/img/architecture.jpg)



## Structure of code
At the root of the project, you will see:

```text
├── app.py  # for running streamlit app
├── attention.py
├── config.py
├── crf.py
├── data
│   ├── glove.6B.100d.txt   # glove embeddings
│   ├── grouped_test.csv    # preprocessed test file
│   ├── grouped_train.csv   # preprocessed train file
│   ├── grouped_val.csv     # preprocessed validation file
│   ├── lookup.pkl          
│   ├── result.csv          # results from the model
│   └── train.csv           # raw converted brat to CoNll train file
├── data_preprocessing.py   # helps to create lookup table and preprocess train,test and val file
├── dataset.py
├── Dockerfile
├── inference.py            # command line inference file
├── main.py                 # Model Trainer file
├── model.py
├── requirements_cpu.txt
├── requirements_gpu.txt
├── savedModels             # folder for saving the trained models
├── train.py
└── utils.py
```
## Run Locally

- Convert the raw brat standoff dataset to CoNLL format.
- Download Converted dataset , glove embeddings , pretrained model from [here](https://drive.google.com/drive/folders/1bKfkFyKidXJJB_q_1o5OZYbg3XILZI6K?usp=share_link) and place it into data folder.
- Modify configuration information in config.py

#### Command Line Inference
```bash
  pip3 install -r requirements_gpu.txt  #if u have gpu
  python3 inference.py      # if you need to take inference on test.csv file
  or
  python3 inference.py -i="Text for inference"
```

#### Streamlit Inference
```bash
  pip3 install -r requirements_gpu.txt  #if u have gpu
  streamlit run app.py
```

#### Docker
```bash
  pip3 install -r requirements_cpu.txt  
  docker build -t streamlit .
  docker run -p 8501:8501 streamlit
```



## Demo
![grab-landing-page](https://github.com/upjabir/lstm_ner/blob/main/img/ner_sama.png)

