#!/usr/bin/env bash

python tensorflow_tasks/test_distributed_tensorflow.py --hosts=localhost:3007,localhost:3008 --job_name=local --task_id=0 & \
python tensorflow_tasks/test_distributed_tensorflow.py --hosts=localhost:3007,localhost:3008 --job_name=local --task_id=1