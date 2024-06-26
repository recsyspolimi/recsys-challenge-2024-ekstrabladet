# RecSysChallenge2024

## Reproduce best submission

### Download the dataset

The first step is to download the dataset ``.parquet`` files and place it in the folder ``dataset/``. In the end you should have a structure as follow:

```plain
├── Ekstra_Bladet_contrastive_vector
│   └── contrastive_vector.parquet
...
├── ebnerd_demo
│   ├── articles.parquet
│   ├── train
│   │   ├── behaviors.parquet
│   │   └── history.parquet
│   └── validation
│       ├── behaviors.parquet
│       └── history.parquet
...
```

### Emotion embeddings

Then, we should run the script to generate emotion embeddings:

```bash
sh ~/RecSysChallenge2024/src/polimi/scripts/run_build_emotion_embedding.sh
```

### Preprocessing

TODO

### Stacking

Our submission is based on stacking. To do so, we need to create the dataframe with the "first-level" model prediction. The procedure is:

1) Train level one models on train set
2) Run inference over the validation set for these models
3) Train level two model using the validation set augmented with models one predictions.

Then, to create the train dataset for the testset:

4) Train level one models on train + validation set
5) Run inference over the testset for these models

Then we can produce previously trained level two models predictions on the testset augmented with level one predictions.

6) Return as final prediction the average predictions of all level two models.


| **Model**            | **Type**     | **Level 1** | **Level 2** |
|----------------------|--------------|-------------|-------------|
| **Catboost**         | Classifier   | *           |  *     |
| **Catboost**         | Ranker       | *           |        |
| **LightGBM**         | Classifier   | *           |  *     |
| **LightGBM**         | Ranker       | *           |        |
| **MLP**              | Classifier   | *           |        |
| **GANDALF**          | Classifier   | *           |        |
| **DEEP & CROSS**     | Classifier   | *           |        |
| **WIDE & DEEP**      | Classifier   | *           |        |


### Training

The following table shows the hyperparameters used for each model and each preprocessing. Neural models have not been trained on the second version of the preprocessing, due to limit of time.

| **Model**            | **Type**     | **Configuration Path**                                  |
|----------------------|--------------|----------------------------------------------------------|
| **Catboost**         | Classifier   | `/path/to/catboost_classifier_config.json`               |
| **Catboost**         | Ranker       | `/path/to/catboost_ranker_config.json`                   |
| **LightGBM**         | Classifier   | `/path/to/lightgbm_classifier_config.json`               |
| **LightGBM**         | Ranker       | `/path/to/lightgbm_ranker_config.json`                   |
| **MLP**              | Classifier   | `/path/to/mlp_classifier_config.json`                    |
| **GANDALF**          | Classifier   | `/path/to/gandalf_classifier_config.json`                |
| **DEEP & CROSS**     | Classifier   | `/path/to/deep_cross_classifier_config.json`             |
| **WIDE & DEEP**      | Classifier   | `/path/to/wide_deep_classifier_config.json`              |

Note that to train each of this model the path of the desired preprocessing version is required, along with the correct configuration file path. Pass them as command line arguments.

Moreover, all models except the ranker have been trained on a subsample of the dataset. To create a subsample of the preprocessing you can run the following script:

```bash
python ~/RecSysChallenge2024/src/polimi/preprocessing_pipelines/subsample_train.py \
     -output_dir ~/RecSysChallenge2024/experiments/ \
     -dataset_dir ~/RecSysChallenge2024/preprocessing/... \
     -original_path ~/RecSysChallenge2024/dataset/ebnerd_small/train/behaviors.parquet
```

where `-dataset_dir` contains the path of the directory with the `train_ds.parquet` preprocessing file.

### Catboost classifier

```bash
python ~/RecSysChallenge2024/src/polimi/scripts/catboost_training.py \
    -output_dir  ~/RecSysChallenge2024/experiments/models \
    -dataset_path ~/RecSysChallenge2024/preprocessing/... \
    -catboost_params_file ~/RecSysChallenge2024/configuration_files/catboost_classifier_recsys_best.json \
    -catboost_verbosity 20 \
    -model_name catboost_classifier
```


### Ranker models

In our solution Catboost ranker has been trained in batches due to memory limitations, an example of the used procedure can be found `~/RecSysChallenge2024/src/polimi/scripts/catboost_ranker_batch_training` inside that folder there's a file named `_procedure_batch_training.txt` that explains the procedure. 

If there are no memory constraint, you can train LightGBM and Catboost ranker by using the same script for the classifier described above by passing the argument `--ranker`.



