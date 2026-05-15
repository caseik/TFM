split_manager
   |
   +--> 1. prepare_split(dataset, experiment_config)
   |         - receives the full dataset and one experiment dictionary
   |         - initializes deterministic random sampling using a fixed seed
   |         - interprets split directives (train, val, test, lora)
   |         - supports fixed counts, zero allocation, and remainder allocation
   |
   +--> 2. select_subsets()
   |         - performs balanced sampling across classes
   |         - randomly selects samples per class
   |         - guarantees reproducibility across runs
   |
   +--> 3. apply_augmentation()
   |         - checks augmentation flag in experiment configuration
   |         - applies the project-wide augmentation protocol to training samples
   |         - leaves validation and test subsets unchanged
   |
   +--> 4. build_split_bundle()
             - packages all generated subsets
             - returns runner-ready data partitions
             - includes train, val, test, and optional lora subsets