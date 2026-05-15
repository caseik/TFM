experiment_logger
   |
   +--> 1. register_batch(experiments)
   |         - receives list of experiment dictionaries
   |         - stores selected experimental parameters
   |         - keeps records in memory
   |
   +--> 2. format_records()
   |         - extracts relevant fields
   |         - organizes rows for publication table
   |
   +--> 3. export_table(path)
             - generates Table 1
             - returns or saves formatted table