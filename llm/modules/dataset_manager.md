main.py
   |
   v
dataset_manager.load_dataset()
   |
   +--> 1. ensure_dataset_available()
   |         - delega en KaggleHub la gestión del dataset
   |         - KaggleHub comprueba cache local (~/.cache)
   |         - si no existe, descarga automáticamente HAM10000
   |
   +--> 2. load_metadata()
   |         - carga HAM10000_metadata.csv
   |
   +--> 3. resolve_image_paths()
   |         - asocia cada image_id con su jpg
   |
   +--> 4. validate_dataset()
   |         - comprueba imágenes faltantes
   |         - comprueba duplicados/inconsistencias
   |
   +--> 5. enrich_metadata()
   |         - añade readable_label
   |         - añade class_idx
   |         - elimina columnas clínicas no utilizadas (age, sex, location, dx_type)
   |
   +--> 6. return DataFrame
             listo para split_manager