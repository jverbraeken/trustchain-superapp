package nl.tudelft.trustchain.fedml.ai

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator
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
import java.util.*

abstract class Runner {
    private var trainDatasetIterator: DataSetIterator? = null
    private var testDatasetIterator: DataSetIterator? = null
    open val seed = Random(System.currentTimeMillis()).nextInt()
    open val printScoreIterations = 5

    fun generateNetwork(
        dataset: Datasets,
        updater: Updaters,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations
    ): MultiLayerNetwork {
        val network = MultiLayerNetwork(
            when (dataset) {
                Datasets.MNIST -> generateDefaultMNISTConfiguration(
                    updater,
                    learningRate,
                    l2
                )
                Datasets.CIFAR10 -> generateDefaultCIFARConfiguration(
                    updater,
                    learningRate,
                    l2
                )
                Datasets.TINYIMAGENET -> generateDefaultTinyImageNetConfiguration(
                    updater,
                    learningRate,
                    l2
                )
            }
        )
        network.init()
        print(momentum)
        return network
    }

    fun generateDefaultMNISTConfiguration(
        updater: Updaters,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        val updaterImpl: IUpdater = when (updater) {
            Updaters.NESTEROVS -> Nesterovs(learningRate.schedule)
            Updaters.ADAM -> Adam(learningRate.schedule)
            Updaters.SGD -> Sgd(learningRate.schedule)
            Updaters.RMSPROP -> RmsProp(learningRate.schedule)
        }
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(updaterImpl)
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
        updater: Updaters,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(getUpdaterImplementation(updater, learningRate))
            .list()



            .layer(0,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(32)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(1,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(2,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(3,
                SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build()
            )
            .layer(4,
                ConvolutionLayer.Builder(intArrayOf(3, 3), intArrayOf(1, 1))
                    .nOut(64)
                    .activation(Activation.RELU)
                    .build()
            )
            .layer(5, DenseLayer.Builder().nOut(64).activation(Activation.RELU).build())
            .layer(6,
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

    fun generateDefaultTinyImageNetConfiguration(
        updater: Updaters,
        learningRate: LearningRates,
        l2: L2Regularizations
    ): MultiLayerConfiguration {
        val updaterImpl: IUpdater = when (updater) {
            Updaters.NESTEROVS -> Nesterovs(learningRate.schedule)
            Updaters.ADAM -> Adam(learningRate.schedule)
            Updaters.SGD -> Sgd(learningRate.schedule)
            Updaters.RMSPROP -> RmsProp(learningRate.schedule)
        }
        return NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(l2.value)
            .weightInit(WeightInit.XAVIER)
            .updater(updaterImpl)
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

    private fun getUpdaterImplementation(updater: Updaters, learningRate: LearningRates): IUpdater {
        return when (updater) {
            Updaters.NESTEROVS -> Nesterovs(learningRate.schedule)
            Updaters.ADAM -> Adam(learningRate.schedule)
            Updaters.SGD -> Sgd(learningRate.schedule)
            Updaters.RMSPROP -> RmsProp(learningRate.schedule)
        }
    }

    fun getTrainDatasetIterator(dataset: Datasets, batchSize: BatchSizes): DataSetIterator {
        if (trainDatasetIterator == null) {
            trainDatasetIterator = when (dataset) {
                Datasets.MNIST -> MnistDataSetIterator(batchSize.value, true, seed)
                Datasets.CIFAR10 -> Cifar10DataSetIterator(
                    batchSize.value,
                    intArrayOf(32, 32),
                    DataSetType.TRAIN,
                    null,
                    seed.toLong()
                )
                Datasets.TINYIMAGENET -> TinyImageNetDataSetIterator(
                    batchSize.value,
                    intArrayOf(32, 32),
                    DataSetType.TRAIN,
                    null,
                    seed.toLong()
                )
            }
        }
        return trainDatasetIterator!!
    }

    fun getTestDatasetIterator(dataset: Datasets, batchSize: BatchSizes): DataSetIterator {
        if (testDatasetIterator == null) {
            testDatasetIterator = when (dataset) {
                Datasets.MNIST -> MnistDataSetIterator(batchSize.value, false, seed)
                Datasets.CIFAR10 -> {
                    val iterator = Cifar10DataSetIterator(
                        batchSize.value,
                        intArrayOf(32, 32),
                        DataSetType.TEST,
                        null,
                        seed.toLong()
                    )
//                    iterator.preProcessor = ImagePreProcessingScaler(-0.5, 0.5)
                    // Should be perhaps NormalizerStandardize (make sure to fit!!!)
                    iterator
                }
                Datasets.TINYIMAGENET -> TinyImageNetDataSetIterator(
                    batchSize.value,
                    intArrayOf(32, 32),
                    DataSetType.TEST,
                    null,
                    seed.toLong()
                )
            }
        }
        return testDatasetIterator!!
    }

    abstract fun run(
        dataset: Datasets,
        updater: Updaters,
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
