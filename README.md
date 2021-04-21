### Movie recommendation system on MovieLens dataset

Dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset

```
├───data
│   ├───pretrained_model    # roberta-base
│   ├───tokenizer           # roberta-base
│   ├───processed
|   └───the-movie-dataset
│
├───data_loader
│   ├───rating_loader.py
│   ├───movie_loader.py
│   └───user_emb_loader_loader.py
│
│
├───model
│   ├───batch.py
│   ├───config.py
│   ├───metric.py
│   └───model.py
│
│
├───saved
│   ├───1st_train      
│       ├───config
│       |───log
│       └───model
│   └───2nd_train
│       ├───config
│       |───log
│       └───model
│
│
├───utils
│   ├───eda_utils.py
│   └───utils.py
│
├───user_based_config.json  # training user based config
├───train_user_based.py 
├───test_user_based.py 
│
│
├───movie_emb_config.json  # training movie embeddings config
├───train_movie_emb.py 
│
│
├───user_emb_config.json   # training user based config
├───train_user_emb.py 
├───test_user_emb.py 
│
│
├───requirements.txt
├───setup.py
└───settings.py
```

## Setup
* Create virtual enviroment
* pip install -r requirements.txt

## Train
```
    python train_user_based.py --config user_based_config.json
    python train_movie_emb.py --config movie_emb_config.json
    python train_user_emb.py --config user_emb_config.json
```

## Test
```
    python test_user_based.py --save_path your_saved_dir_in_config.json
    python test_user_emb.py --save_path your_saved_dir_in_config.json
```

## TensorBoard
```
    tensorboard --logdir your_saved_dir_in_config.json
```
