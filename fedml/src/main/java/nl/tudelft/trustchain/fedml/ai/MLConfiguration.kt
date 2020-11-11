package nl.tudelft.trustchain.fedml.ai

import nl.tudelft.trustchain.fedml.*

data class MLConfiguration(
    val dataset: Datasets,
    val datasetIteratorConfiguration: DatasetIteratorConfiguration,
    val nnConfiguration: NNConfiguration,
    val trainConfiguration: TrainConfiguration,
    val modelPoisoningConfiguration: ModelPoisoningConfiguration
)

data class DatasetIteratorConfiguration(
    val batchSize: BatchSizes,
    val distribution: IteratorDistributions,
    val maxTestSamples: MaxTestSamples?
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

data class ModelPoisoningConfiguration(
    val attack: ModelPoisoningAttacks,
    val numAttackers: NumAttackers
)
