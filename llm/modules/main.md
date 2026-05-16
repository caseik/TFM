main.py
   |
   +--> 1. dataset_manager.load_dataset()
   |         - loads HAM10000 metadata + image paths
   |         - validates and enriches dataset metadata
   |
   +--> 2. build_experiment_list()
   |         - defines all planned experiments
   |         - each experiment is represented as a Python dict
   |
   +--> 3. experiment_logger.register_batch()
   |         - registers the full experimental plan
   |
   +--> 4. experiment_logger.export_table()
   |         - generates Table 1 before execution
   |
   +--> 5. run_experiments()
             |
             +--> iterates over experiments
             |
             +--> split_manager.prepare_split()
             |         - creates experiment-specific splits
             |         - applies balanced sampling
             |         - applies optional augmentation
             |
             +--> dispatch to model runner
NOT COMPLETE