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
import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.math.BigInteger
import kotlin.random.Random

private val logger = KotlinLogging.logger("Runner")

private const val TEST_SET_SIZE = 10

fun generateDefaultMNISTConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .l2(nnConfiguration.l2.value)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(10)
            .build()
        )
        .layer(SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build()
        )
        .layer(ConvolutionLayer
            .Builder(intArrayOf(5, 5), intArrayOf(1, 1))
            .nOut(50)
            .build()
        )
        .layer(SubsamplingLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(2, 2), intArrayOf(2, 2))
            .build()
        )
        .layer(DenseLayer
            .Builder()
            .nOut(500)
            .build()
        )
        .layer(OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(10)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
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
        .seed(seed.toLong())
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list()
        .layer(ConvolutionLayer
            .Builder(intArrayOf(3, 3), intArrayOf(1, 1), intArrayOf(1, 1))
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
            .nOut(16)
            .build()
        )
        .layer(BatchNormalization())
        .layer(SubsamplingLayer
            .Builder(intArrayOf(3, 3), intArrayOf(1, 1), intArrayOf(1, 1))
            .poolingType(SubsamplingLayer.PoolingType.MAX)
            .build()
        )
        .layer(BatchNormalization())
        .layer(SubsamplingLayer
            .Builder(intArrayOf(2, 2), intArrayOf(2, 2))
            .poolingType(SubsamplingLayer.PoolingType.MAX)
            .build()
        )

        .layer(OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .dropOut(0.8)
            .nOut(numLabels)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build()
        )
        .setInputType(InputType.convolutional(height.toLong(), width.toLong(), channels.toLong()))
        .build()
}

fun generateDefaultHARConfiguration(
    nnConfiguration: NNConfiguration,
    seed: Int,
): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder()
        .seed(seed.toLong())
        .l2(nnConfiguration.l2.value)
        .activation(Activation.LEAKYRELU)
        .weightInit(WeightInit.RELU)
        .updater(nnConfiguration.optimizer.inst(nnConfiguration.learningRate))
        .list()
        .layer(Convolution1DLayer
            .Builder(3)
            .nIn(128)
            .nOut(64)
            .build()
        )
        .layer(Subsampling1DLayer
            .Builder(SubsamplingLayer.PoolingType.MAX, 2, 2)
            .build()
        )
        .layer(Convolution1DLayer
            .Builder(3)
            .nOut(64)
            .build()
        )
        .layer(GlobalPoolingLayer
            .Builder(PoolingType.MAX)
            .build()
        )
        .layer(OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .nOut(HARDataFetcher.NUM_LABELS)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build()
        )
        .setInputType(InputType.recurrent(9, 128))
        .build()
}

abstract class Runner {
    protected open val printScoreIterations = 5
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
        val testDataSetIterator = dataset.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_200,
                List(datasetIteratorConfiguration.distribution.size) { TEST_SET_SIZE },
                datasetIteratorConfiguration.maxTestSamples),
            seed + 2,
            CustomDataSetType.TEST,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded testDataSetIterator" }
        val fullTestDataSetIterator = dataset.inst(
            DatasetIteratorConfiguration(BatchSizes.BATCH_200,
                List(datasetIteratorConfiguration.distribution.size) { datasetIteratorConfiguration.maxTestSamples.value },
                datasetIteratorConfiguration.maxTestSamples),
            seed + 3,
            CustomDataSetType.FULL_TEST,
            baseDirectory,
            behavior
        )
        logger.debug { "Loaded fullTestDataSetIterator" }
        return listOf(trainDataSetIterator, testDataSetIterator, fullTestDataSetIterator)
    }

    protected fun craftMessage(first: INDArray, behavior: Behaviors, random: Random): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first, random)
            Behaviors.LABEL_FLIP -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray, random: Random): INDArray {
        val oldMatrix = toFloatArray(first)
        val newVector = FloatArray(oldMatrix.size)
        for (i in oldMatrix.indices) {
            newVector[i] = random.nextFloat() * 2 - 1
        }
        return NDArray(Array(1) { newVector})
    }

    private fun toFloatArray(first: INDArray): FloatArray {
        val data = first.data()
        val length = data.length().toInt()
        val indexer = data.indexer() as FloatRawIndexer
        val array = FloatArray(length)
        for (i in 0 until length) {
            array[i] = indexer.getRaw(i.toLong())
        }
        return array
    }
}
