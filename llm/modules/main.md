main.py
   |
   +--> 1. dataset_manager.load_dataset()
   |         - loads HAM10000 metadata + image paths
   |
   +--> 2. split_manager.create_splits()
   |         - creates patient-aware experimental splits
   |
   +--> 3. build_experiment_list()
   |         - defines all planned experiments
   |         - each experiment is represented as a Python dict
   |
   +--> 4. experiment_logger.register_batch()
   |         - sends experimental plan for documentation
   |
   +--> 5. experiment_logger.export_table()
   |         - generates Table 1 before execution
   |
   +--> 6. run_experiments()
             - dispatches each experiment to its runner

NOT COMPLETE