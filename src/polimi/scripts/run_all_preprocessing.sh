#!/bin/bash


python3 ~/RecSysChallenge2024/src/polimi/scripts/create_embeddings_folder.py && echo "create_embeddings_folder.py completed successfully."

sh ~/RecSysChallenge2024/src/polimi/scripts/run_build_emotion_embedding.sh && echo "run_build_emotion_embedding.sh completed successfully."

sh ~/RecSysChallenge2024/src/polimi/scripts/run_build_kenneth_embedding.sh && echo "run_build_kenneth_embedding.sh completed successfully."

sh ~/RecSysChallenge2024/src/polimi/scripts/run_build_distil_embedding.sh && echo "run_build_distil_embedding.sh completed successfully."

sh ~/RecSysChallenge2024/src/polimi/scripts/run_preprocessing.sh && echo "run_preprocessing.sh completed successfully."

sh ~/RecSysChallenge2024/src/polimi/scripts/run_add_recsys_features.sh && echo "run_add_recsys_features.sh completed successfully."