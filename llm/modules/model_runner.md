model runners
   |
   +--> MLPRunner.run(split_bundle)
   |         - validate embedding inputs
   |         - prepare tensors
   |         - train MLP
   |         - run inference
   |         - package predictions
   |
   +--> CentroidRunner.run(split_bundle)
   |         - validate embedding inputs
   |         - compute class centroids
   |         - compute distances
   |         - assign predictions
   |         - package predictions
   |
   +--> ResNetRunner.run(split_bundle)
             - validate image inputs
             - create dataloaders
             - train CNN
             - run inference
             - package predictions