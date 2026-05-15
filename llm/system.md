                         +----------------------+
                         |       main.py        |
                         |  Orquestador global  |
                         +----------+-----------+
                                    |
                                    v
                    +---------------+----------------+
                    |                                |
                    v                                v
        +----------------------+       +----------------------+
        |   dataset_manager    |       |   experiment_logger  |
        |----------------------|       |----------------------|
        | - download/cache     |       | Tabla 1 (input exp.) |
        | - load HAM10000      |       | config experimental  |
        +----------+-----------+       +----------------------+
                   |
                   v
        +----------------------+
        |    split_manager     |
        |----------------------|
        | - create_splits()    |
        | - stratified split   |
        | - patient split      |
        +----------+-----------+
                   |
                   v
      +------------+--------------------------------------+
      |                model runners                      |
      |---------------------------------------------------|
      | embeddings_runner.py                              |
      | finetuning_runner.py                              |
      | lora_runner.py                                    |
      +----------------------+----------------------------+
                             |
                             v
                +------------+----------------+
                |      metrics_generator      |
                |-----------------------------|
                | Tabla 2 (accuracy/clases)  |
                | Tabla 3 (FN clínicos)      |
                +-----------------------------+


main.py is responsible for constructing the full experimental plan after dataset loading and split generation. Each experiment specification is registered through the logger before execution, allowing Table 1 to be generated prior to model training.