package nl.tudelft.trustchain.fedml.ai

data class MLConfiguration(
    val dataset: Datasets,
    val datasetIteratorConfiguration: DatasetIteratorConfiguration,
    val nnConfiguration: NNConfiguration,
    val trainConfiguration: TrainConfiguration
)

data class DatasetIteratorConfiguration(
    val batchSize: BatchSizes,
    val distribution: IteratorDistributions,
    val maxTestSamples: MaxSamples?
)

data class NNConfiguration(
    val optimizer: Optimizers,
    val learningRate: LearningRates,
    val momentum: Momentums?,
    val l2: L2Regularizations
)

data class TrainConfiguration(
    val numEpochs: Epochs,
    val gar: GARs,
    val communicationPattern: CommunicationPatterns,
    val behavior: Behaviors
)
