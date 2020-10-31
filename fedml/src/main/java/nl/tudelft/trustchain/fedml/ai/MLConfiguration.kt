package nl.tudelft.trustchain.fedml.ai

data class MLConfiguration(
    val dataset: Datasets,
    val optimizer: Optimizers,
    val learningRate: LearningRates,
    val momentum: Momentums?,
    val l2: L2Regularizations,
    val batchSize: BatchSizes,
    val epoch: Epochs,
    val iteratorDistribution: IteratorDistributions,
    val maxTestSamples: MaxTestSamples
)
