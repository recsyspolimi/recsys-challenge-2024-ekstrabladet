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
