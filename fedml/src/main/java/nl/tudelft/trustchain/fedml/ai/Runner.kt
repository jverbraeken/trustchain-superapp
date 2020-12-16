package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.BatchSizes
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.Datasets
import nl.tudelft.trustchain.fedml.MaxTestSamples
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.har.HARDataFetcher
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.learning.config.AdaDelta
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.math.BigInteger
import kotlin.random.Random

private val logger = KotlinLogging.logger("Runner")

fun generateDefaultMNISTConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .l2(nnConfiguration.l2.value)
        .weightInit(WeightInit.XAVIER)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            ConvolutionLayer.Builder(5, 5)
//                .nIn(1)
                .stride(1, 1)
                .nOut(10)
                .activation(Activation.IDENTITY)
                .build()
        )
        .layer(
            SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build()
        )
        .layer(
            ConvolutionLayer.Builder(5, 5)
                .stride(1, 1)
                .nOut(50)
                .activation(Activation.IDENTITY)
                .build()
        )
        .layer(
            SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build()
        )
        .layer(
            DenseLayer.Builder()
                .activation(Activation.RELU)
                .nOut(512)
                .build()
        )
        .layer(
            OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .build()
        )
        .setInputType(InputType.convolutionalFlat(28, 28, 1))
        .build()
}

fun generateDefaultCIFARConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
): MultiLayerConfiguration {
    val width = CifarLoader.WIDTH
    val height = CifarLoader.HEIGHT
    val channels = CifarLoader.CHANNELS
    val numLabels = CifarLoader.NUM_LABELS
    return NeuralNetConfiguration.Builder()
        .seed(123L)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .weightInit(WeightInit.XAVIER)
        .list()
        .layer(ConvolutionLayer
                .Builder(intArrayOf(3, 3), intArrayOf(1, 1), intArrayOf(1, 1))
                .activation(Activation.LEAKYRELU)
                .nIn(CifarLoader.CHANNELS)
                .nOut(32)
                .build()
        )
        .layer(BatchNormalization())
        .layer(SubsamplingLayer
            .Builder(intArrayOf(2, 2), intArrayOf(2, 2))
            .poolingType(SubsamplingLayer.PoolingType.MAX)
            .build()
        )

        .layer(ConvolutionLayer
            .Builder(intArrayOf(1, 1), intArrayOf(1, 1), intArrayOf(1, 1))
            .activation(Activation.LEAKYRELU)
            .nOut(16)
            .build()
        )
        .layer(BatchNormalization())
        .layer(SubsamplingLayer
            .Builder(intArrayOf(2, 2), intArrayOf(2, 2))
            .poolingType(SubsamplingLayer.PoolingType.MAX)
            .build()
        )

        .layer(OutputLayer . Builder (LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numLabels)
            .dropOut(0.8)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutional(height.toLong(), width.toLong(), channels.toLong()))
        .build();
}

fun generateDefaultHARConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .l2(nnConfiguration.l2.value)
        .weightInit(WeightInit.XAVIER)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            Convolution1DLayer.Builder(3).nIn(128).activation(Activation.RELU).nOut(64).build()
        )
        .layer(Convolution1DLayer.Builder(3).activation(Activation.RELU).nOut(64).build())
        .layer(DropoutLayer.Builder(0.5).nOut(64).build())
        .layer(GlobalPoolingLayer.Builder(PoolingType.MAX).build())
        .layer(
            OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(HARDataFetcher.NUM_LABELS)
                .activation(Activation.SOFTMAX)
                .build()
        )
        .setInputType(InputType.recurrent(9, 128))
        .build()
}

abstract class Runner {
    protected open val printScoreIterations = 5
    protected val iterationsBeforeEvaluation = 15
    protected val iterationsBeforeSending = iterationsBeforeEvaluation * 2
    protected val bigPrime = BigInteger("100012421")
    protected val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    fun generateNetwork(
        dataset: Datasets,
        nnConfiguration: NNConfiguration,
        seed: Int,
    ): MultiLayerNetwork {
        val network = MultiLayerNetwork(dataset.architecture(nnConfiguration, seed))
        network.init()
        return network
    }

    abstract fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    )

    protected fun getDataSetIterators(
        dataset: Datasets,
        datasetIteratorConfiguration: DatasetIteratorConfiguration,
        seed: Long,
        baseDirectory: File,
        behavior: Behaviors,
    ): List<CustomDataSetIterator> {
        val trainDataSetIterator = dataset.inst(
            DatasetIteratorConfiguration(datasetIteratorConfiguration.batchSize,
                datasetIteratorConfiguration.distribution,
                datasetIteratorConfiguration.maxTestSamples),
            seed,
            CustomDataSetType.TRAIN,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded trainDataSetIterator" }
        val fullTrainDataSetIterator = dataset.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                listOf(5, 5, 5, 5, 5, 5, 5, 5, 5, 5),
                MaxTestSamples.NUM_20),
            seed,
            CustomDataSetType.TRAIN,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded fullTrainDataSetIterator" }
        val testDataSetIterator = dataset.inst(
            datasetIteratorConfiguration,
            seed,
            CustomDataSetType.TEST,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded testDataSetIterator" }
        val fullTestDataSetIterator = dataset.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_5,
                List(datasetIteratorConfiguration.distribution.size) { datasetIteratorConfiguration.maxTestSamples.value },
                datasetIteratorConfiguration.maxTestSamples),
            seed,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded fullTestDataSetIterator" }
        return listOf(trainDataSetIterator, fullTrainDataSetIterator, testDataSetIterator, fullTestDataSetIterator)
    }

    protected fun craftMessage(first: INDArray, behavior: Behaviors, random: Random): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first, random)
            Behaviors.LABEL_FLIP -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val oldMatrix = first.toFloatMatrix()[0]
        val newMatrix = Array(1) { FloatArray(oldMatrix.size) }
        for (i in oldMatrix.indices) {
            newMatrix[0][i] = random.nextFloat() * 2 - 1
        }
        return NDArray(newMatrix)
    }
}
