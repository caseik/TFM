feature_extractor
   |
   +--> 1. load_backbone(config)
   |         - loads embedding model (DINOv2, CLIP, ...)
   |         - prepares preprocessing pipeline
   |         - returns inference-ready extractor
   |
   +--> 2. build_cache_key(split, config)
   |         - builds unique embedding signature
   |         - includes backbone + preprocessing + split identity
   |         - returns cache identifier
   |
   +--> 3. transform_split_bundle(split_bundle)
   |         - iterates over all splits
   |         - checks cache before inference
   |         - computes missing embeddings only
   |
   +--> 4. save_embeddings(...)
   |         - persists computed embeddings
   |         - stores metadata alignment
   |
   +--> 5. load_embeddings(...)
             - restores cached embeddings
             - reconstructs split outputs