package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.BatchSizes
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.cifar.CustomCifar10Fetcher
import nl.tudelft.trustchain.fedml.ai.dataset.har.HARDataFetcher
import nl.tudelft.trustchain.fedml.ai.dataset.mobi_act.MobiActDataFetcher
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.math.BigInteger
import kotlin.math.min
import kotlin.random.Random

private val logger = KotlinLogging.logger("Runner")

private const val TEST_SET_SIZE = 10

enum class NNConfigurationMode {
    REGULAR, TRANSFER, FROZEN
}

fun generateDefaultMNISTConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val numClasses = if (mode == NNConfigurationMode.TRANSFER) 26 else 10
    val layers = arrayOf<Layer>(
        ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(10)
            .build(),
        SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build(),
        ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(50)
            .build(),
        SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build(),
        OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(numClasses)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .hasBias(false)
            .build()
    )
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .l2(nnConfiguration.l2.value)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[0]).build()
            } else {
                layers[0]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[1]).build()
            } else {
                layers[1]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[2]).build()
            } else {
                layers[2]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[3]).build()
            } else {
                layers[3]
            }
        )
        .layer(layers[4])
        .setInputType(InputType.convolutionalFlat(28, 28, 1))
        .build()
}

fun generateDefaultCIFARConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val width = CifarLoader.WIDTH
    val height = CifarLoader.HEIGHT
    val channels = CifarLoader.CHANNELS
    val numClasses = if (mode == NNConfigurationMode.TRANSFER) CustomCifar10Fetcher.NUM_LABELS_TRANSFER else CifarLoader.NUM_LABELS
    val layers = arrayOf<Layer>(

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nIn(channels).nOut(64).build(),

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(128).build(),
        SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(),

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(128).build(),
        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(128).build(),

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(256).build(),
        SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(),

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(512).build(),
        SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(),

        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(512).build(),
        ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
            .nOut(512).build(),

        OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numClasses)
            .activation(Activation.SOFTMAX)
            .build()
    )
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list()
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[0]).build()
            } else {
                layers[0]
            }
        )
        .layer(BatchNormalization())

        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[1]).build()
            } else {
                layers[1]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[2]).build()
            } else {
                layers[2]
            }
        )
        .layer(BatchNormalization())

        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[3]).build()
            } else {
                layers[3]
            }
        )
        .layer(BatchNormalization())
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[4]).build()
            } else {
                layers[4]
            }
        )
        .layer(BatchNormalization())

        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[5]).build()
            } else {
                layers[5]
            }
        )
        .layer(BatchNormalization())
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[6]).build()
            } else {
                layers[6]
            }
        )
        .layer(BatchNormalization())

        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[7]).build()
            } else {
                layers[7]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[8]).build()
            } else {
                layers[8]
            }
        )
        .layer(BatchNormalization())

        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[9]).build()
            } else {
                layers[9]
            }
        )
        .layer(BatchNormalization())
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[10]).build()
            } else {
                layers[10]
            }
        )
        .layer(BatchNormalization())

        .layer(layers[11])
        .setInputType(InputType.convolutional(height.toLong(), width.toLong(), channels.toLong()))
        .build()
}

fun generateDefaultHARConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val numClasses = if (mode == NNConfigurationMode.TRANSFER) HARDataFetcher.NUM_LABELS else HARDataFetcher.NUM_LABELS
    val layers = arrayOf<Layer>(
        Convolution1DLayer
            .Builder(3)
            .nIn(128)
            .nOut(64)
            .build(),
        Subsampling1DLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, 2, 2)
            .build(),
        Convolution1DLayer
            .Builder(3)
            .nOut(64)
            .build(),
        GlobalPoolingLayer
            .Builder(PoolingType.MAX)
            .build(),
        OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(HARDataFetcher.NUM_LABELS)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build(),
    )
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .l2(nnConfiguration.l2.value)
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[0]).build()
            } else {
                layers[0]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[1]).build()
            } else {
                layers[1]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[2]).build()
            } else {
                layers[2]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[3]).build()
            } else {
                layers[3]
            }
        )
        .layer(layers[4])
        .setInputType(InputType.recurrent(9, 128))
        .build()
}

fun generateDefaultMobiActConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
    mode: NNConfigurationMode,
): MultiLayerConfiguration {
    val numClasses = if (mode == NNConfigurationMode.TRANSFER) 6 else -1
    val layers = arrayOf<Layer>(
        Convolution1DLayer
            .Builder(5, 1, 1)
            .nIn(3)
            .nOut(64)
            .build(),
        Subsampling1DLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, 2, 2)
            .build(),
        Convolution1DLayer
            .Builder(3, 1, 2)
            .nOut(128)
            .build(),
        Subsampling1DLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, 2, 2)
            .build(),
        Convolution1DLayer
            .Builder(3, 1, 1)
            .nOut(256)
            .build(),
        GlobalPoolingLayer
            .Builder(PoolingType.MAX)
            .build(),
        OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(6)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build(),
    )
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .l2(nnConfiguration.l2.value)
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[0]).build()
            } else {
                layers[0]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[1]).build()
            } else {
                layers[1]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[2]).build()
            } else {
                layers[2]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[3]).build()
            } else {
                layers[3]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[4]).build()
            } else {
                layers[4]
            }
        )
        .layer(
            if (mode == NNConfigurationMode.FROZEN) {
                FrozenLayer.Builder().layer(layers[5]).build()
            } else {
                layers[5]
            }
        )
        .layer(layers[6])
        .setInputType(InputType.recurrent(3, 500))
        .build()
}

abstract class Runner {
    protected open val printScoreIterations = 5
    protected val bigPrime = BigInteger("100012421")
    protected val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    fun generateNetwork(
        architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration,
        nnConfiguration: NNConfiguration,
        seed: Int,
        mode: NNConfigurationMode,
    ): MultiLayerNetwork {
        val network = MultiLayerNetwork(architecture(nnConfiguration, seed, mode))
        network.init()
        return network
    }

    abstract fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    )

    protected fun getDataSetIterators(
        inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDataSetType, baseDirectory: File, behavior: Behaviors, transfer: Boolean) -> CustomDataSetIterator,
        datasetIteratorConfiguration: DatasetIteratorConfiguration,
        seed: Long,
        baseDirectory: File,
        behavior: Behaviors,
    ): List<CustomDataSetIterator> {
        val trainDataSetIterator = inst(
            DatasetIteratorConfiguration(
                datasetIteratorConfiguration.batchSize,
                datasetIteratorConfiguration.distribution,
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed,
            CustomDataSetType.TRAIN,
            baseDirectory,
            behavior,
            false,
        )
        logger.debug { "Loaded trainDataSetIterator" }
        val testDataSetIterator = inst(
            DatasetIteratorConfiguration(
                BatchSizes.BATCH_200,
                datasetIteratorConfiguration.distribution.map { min(TEST_SET_SIZE, it) },
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 1,
            CustomDataSetType.TEST,
            baseDirectory,
            behavior,
            false,
        )
        logger.debug { "Loaded testDataSetIterator" }
        val fullTestDataSetIterator = inst(
            DatasetIteratorConfiguration(
                BatchSizes.BATCH_200,
                List(datasetIteratorConfiguration.distribution.size) { datasetIteratorConfiguration.maxTestSamples.value },
                datasetIteratorConfiguration.maxTestSamples
            ),
            seed + 2,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            behavior,
            false,
        )
        logger.debug { "Loaded fullTestDataSetIterator" }
        return listOf(trainDataSetIterator, testDataSetIterator, fullTestDataSetIterator)
    }

    protected fun craftMessage(first: INDArray, behavior: Behaviors, random: Random): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first, random)
            Behaviors.LABEL_FLIP_2 -> first
            Behaviors.LABEL_FLIP_ALL -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val newVector = FloatArray(first.length().toInt()) { random.nextFloat() / 2 - 0.2f }
        return NDArray(Array(1) { newVector })
    }
}
