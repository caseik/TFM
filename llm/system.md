                         +----------------------+
                         |       main.py        |
                         |  Global orchestrator |
                         +----------+-----------+
                                    |
                                    v
                    +---------------+----------------+
                    |                                |
                    v                                v
        +----------------------+       +----------------------+
        |   dataset_manager    |       |  experiment_logger   |
        |----------------------|       |----------------------|
        | - download/cache     |       | Table 1 (exp. plan)  |
        | - load HAM10000      |       | experimental config  |
        +----------+-----------+       +----------------------+
                   |
                   v
        +----------------------+
        |    split_manager     |
        |----------------------|
        | - create_splits()    |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |  feature_extractor   |
        |----------------------|
        | - optional stage     |
        | - image -> embedding |
        | - skipped if none    |
        +----------+-----------+
                   |
                   v
      +------------+--------------------------------------+
      |                model runners                      |
      |---------------------------------------------------|
      | mlp_runner.py                                     |
      | centroid_runner.py                                |
      | resnet_runner.py                                  |
      | lora_runner.py                                    |
      +----------------------+----------------------------+
                             |
                             v
                +------------+----------------+
                |    metrics_generator        |
                |-----------------------------|
                | Table 2 (accuracy/classes) |
                | Table 3 (clinical FN)      |
                +-----------------------------+


main.py is responsible for constructing the full experimental plan after dataset loading and split generation. Each experiment specification is registered through the logger before execution, allowing Table 1 to be generated prior to model training.