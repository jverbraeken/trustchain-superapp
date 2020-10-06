package nl.tudelft.trustchain.fedml.ai

import nl.tudelft.trustchain.fedml.ai.dataset.har.HARDataFetcher
import nl.tudelft.trustchain.fedml.ai.dataset.har.HARIterator
import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.*
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.util.*

abstract class Runner {
    private var trainDatasetIterator: DataSetIterator? = null
    private var testDatasetIterator: DataSetIterator? = null
    open val seed = Random(System.currentTimeMillis()).nextInt()
    open val printScoreIterations = 5

    fun generateNetwork(
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations
    ): MultiLayerNetwork {
        val network =
            MultiLayerNetwork(dataset.defaultArchitecture(this, optimizer, learningRate, l2))
        network.init()
        print(momentum)
        return network
    }

    fun generateDefaultMNISTConfiguration(
        optimizer: Optimizers,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(getOptimizerImplementation(optimizer, learningRate))
            .list()
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .nIn(1)
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
                    .nOut(500)
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
        optimizer: Optimizers,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(getOptimizerImplementation(optimizer, learningRate))
            .list()


            .layer(
                0,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(32)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(
                1,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                2,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(
                3,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(
                4,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(5, DenseLayer.Builder().nOut(64).activation(Activation.RELU).build())
            .layer(
                6,
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(CifarLoader.NUM_LABELS)
                    .activation(Activation.SOFTMAX)
                    .build()
            )


            /*.layer(0,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(32)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(1, BatchNormalization.Builder().nOut(32).build())
            .layer(2,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(32)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(3, BatchNormalization.Builder().nOut(32).build())
            .layer(4,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(5, DropoutLayer.Builder(0.2).build())

            .layer(6,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(7, BatchNormalization.Builder().nOut(64).build())
            .layer(8,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(9, BatchNormalization.Builder().nOut(64).build())
            .layer(10,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(11, DropoutLayer.Builder(0.3).build())

            .layer(12,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(128)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(13, BatchNormalization.Builder().nOut(128).build())*//*
            .layer(14,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(128)
                    .l2(l2.value)
                    .activation(Activation.ELU)
                    .build()
            )
            .layer(15, BatchNormalization.Builder().nOut(128).build())*//*
            .layer(14,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(15, DropoutLayer.Builder(0.4).build())

            .layer(16,
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(CifarLoader.NUM_LABELS)
                    .activation(Activation.SOFTMAX)
                    .build()
            )*/
            .setInputType(InputType.convolutional(32, 32, 3))
            .build()
    }

    private fun convInit(
        name: String,
        `in`: Int,
        out: Int,
        kernel: IntArray,
        stride: IntArray,
        pad: IntArray,
        bias: Double
    ): ConvolutionLayer {
        return ConvolutionLayer.Builder(kernel, stride, pad)
            .name(name).nIn(`in`).nOut(out).biasInit(bias).build()
    }

    private fun conv3x3(
        name: String,
        out: Int,
        bias: Double
    ): ConvolutionLayer {
        return ConvolutionLayer.Builder(
            intArrayOf(3, 3),
            intArrayOf(1, 1),
            intArrayOf(1, 1)
        ).name(name).nOut(out).biasInit(bias).build()
    }

    private fun conv5x5(
        name: String,
        out: Int,
        stride: IntArray,
        pad: IntArray,
        bias: Double
    ): ConvolutionLayer {
        return ConvolutionLayer.Builder(
            intArrayOf(5, 5),
            stride,
            pad
        ).name(name).nOut(out).biasInit(bias).build()
    }

    private fun maxPool(
        name: String,
        kernel: IntArray
    ): SubsamplingLayer {
        return SubsamplingLayer.Builder(kernel, intArrayOf(2, 2))
            .name(name).build()
    }

    private fun fullyConnected(
        name: String,
        out: Int,
        bias: Double,
        dropOut: Double
    ): DenseLayer {
        return DenseLayer.Builder().name(name).nOut(out)
            .biasInit(bias).dropOut(dropOut).build()
    }

    fun generateDefaultTinyImageNetConfiguration(
        optimizer: Optimizers,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.RELU)
            .convolutionMode(ConvolutionMode.Same)
            .updater(getOptimizerImplementation(optimizer, learningRate))
            .list()
            .layer(
                0,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(32)
                    .activation(Activation.RELU)
                    .build()
            )/*
            .layer(
                0,
                convInit(
                    "cnn1",
                    TinyImageNetFetcher.INPUT_CHANNELS,
                    96,
                    intArrayOf(11, 11),
                    intArrayOf(4, 4),
                    intArrayOf(3, 3),
                    0.0
                )
            )*/
            /*.layer(1, LocalResponseNormalization.Builder().name("lrn1").build())
            .layer(2, maxPool("maxpool1", intArrayOf(3, 3)))
            .layer(3, conv5x5("cnn2", 256, intArrayOf(1, 1), intArrayOf(2, 2), 1.0))
            .layer(4, LocalResponseNormalization.Builder().name("lrn2").build())
            .layer(5, maxPool("maxpool2", intArrayOf(3, 3)))
            .layer(6, conv3x3("cnn3", 384, 0.0))
            .layer(7, conv3x3("cnn4", 384, 1.0))
            .layer(8, conv3x3("cnn5", 256, 1.0))
            .layer(9, maxPool("maxpool3", intArrayOf(3, 3)))
            .layer(10, fullyConnected("ffn1", 4096, 1.0, 0.5))
            .layer(11, fullyConnected("ffn2", 4096, 1.0, 0.5))*/
            .layer(
                1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .name("output")
                    .nOut(TinyImageNetFetcher.NUM_LABELS)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(
                InputType.convolutional(
                    TinyImageNetFetcher.INPUT_HEIGHT.toLong(),
                    TinyImageNetFetcher.INPUT_WIDTH.toLong(),
                    TinyImageNetFetcher.INPUT_CHANNELS.toLong()
                )
            )
            .build();
    }

    fun generateDefaultHALConfiguration(
        optimizer: Optimizers,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(getOptimizerImplementation(optimizer, learningRate))
            .list()
            .layer(Convolution1DLayer.Builder(3).nIn(128).activation(Activation.RELU).nOut(64).build())
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

    private fun getOptimizerImplementation(optimizer: Optimizers, learningRate: LearningRates): IUpdater {
        return when (optimizer) {
            Optimizers.NESTEROVS -> Nesterovs(learningRate.schedule)
            Optimizers.ADAM -> Adam(learningRate.schedule)
            Optimizers.SGD -> Sgd(learningRate.schedule)
            Optimizers.RMSPROP -> RmsProp(learningRate.schedule)
            Optimizers.AMSGRAD -> AMSGrad(learningRate.schedule)
        }
    }

    fun getTrainDatasetIterator(
        baseDirectory: File,
        dataset: Datasets,
        batchSize: BatchSizes
    ): DataSetIterator {
        if (trainDatasetIterator == null) {
            trainDatasetIterator = when (dataset) {
                Datasets.MNIST -> MnistDataSetIterator(batchSize.value, true, seed)
                Datasets.CIFAR10 -> Cifar10DataSetIterator(
                    batchSize.value,
                    null,
                    DataSetType.TRAIN,
                    null,
                    seed.toLong()
                )
                Datasets.TINYIMAGENET -> TinyImageNetDataSetIterator(
                    batchSize.value,
                    null,
                    DataSetType.TRAIN,
                    null,
                    seed.toLong()
                )
                Datasets.HAL -> {
                    val iterator =
                        HARIterator(
                            baseDirectory,
                            batchSize.value,
                            true,
                            seed
                        )
                    iterator
                }
            }
        }
        return trainDatasetIterator!!
    }

    fun getTestDatasetIterator(
        baseDirectory: File,
        dataset: Datasets,
        batchSize: BatchSizes
    ): DataSetIterator {
        if (testDatasetIterator == null) {
            testDatasetIterator = when (dataset) {
                Datasets.MNIST -> MnistDataSetIterator(batchSize.value, false, seed)
                Datasets.CIFAR10 -> {
                    val iterator = Cifar10DataSetIterator(
                        batchSize.value,
                        null,
                        DataSetType.TEST,
                        null,
                        seed.toLong()
                    )
                    iterator.preProcessor = ImagePreProcessingScaler(-0.5, 0.5)
                    // Should be perhaps NormalizerStandardize (make sure to fit!!!)
                    iterator
                }
                Datasets.TINYIMAGENET -> {
                    val iterator = TinyImageNetDataSetIterator(
                        batchSize.value,
                        null,
                        DataSetType.TEST,
                        null,
                        seed.toLong()
                    )
                    iterator.preProcessor = ImagePreProcessingScaler(-0.5, 0.5)
                    iterator
                }
                Datasets.HAL -> {
                    val iterator =
                        HARIterator(
                            baseDirectory,
                            batchSize.value,
                            false,
                            seed
                        )
                    iterator
                }
            }
        }
        return testDatasetIterator!!
    }

    abstract fun run(
        baseDirectory: File,
        numEpochs: Epochs,
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    )

    fun calculateWeightedAverageParams(params: List<Pair<INDArray, Int>>): Pair<INDArray, Int> {
        val totalWeight = params.map { it.second }.reduce { sum, elem -> sum + elem }
        var arr: INDArray =
            params[0].first.mul(params[0].second.toDouble() / totalWeight.toDouble())
        for (i in 1 until params.size) {
            arr = arr.add(params[i].first.mul(params[i].second.toDouble() / totalWeight.toDouble()))
        }
        return Pair(arr, totalWeight)
    }
}
