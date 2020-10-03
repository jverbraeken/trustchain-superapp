package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import kotlin.random.Random

abstract class MNISTRunner {
    open val nChannels = 1
    open val outputNum = 10
    open val batchSize = 64
    open val seed = 1 //Random(System.currentTimeMillis()).nextInt()
    open val printScoreIterations = 20

    val mnistTrain by lazy {
        MnistDataSetIterator(batchSize, true, seed)
    }

    val mnistTest by lazy {
        MnistDataSetIterator(batchSize, false, seed)
    }

    //    val learningRateSchedule by lazy {
//        val map: MutableMap<Int, Double> = HashMap()
//        map[0] = 0.06
//        map[200] = 0.05
//        map[600] = 0.028
//        map[800] = 0.0060
//        map[1000] = 0.001
//        map
//    }
    val nnConf: MultiLayerConfiguration by lazy {
        NeuralNetConfiguration.Builder()
            .seed(seed.toLong())
            .l2(0.0005)
            .weightInit(WeightInit.XAVIER)
            .updater(Adam(1e-3)/*Nesterovs(MapSchedule(ScheduleType.ITERATION, learningRateSchedule))*/)
            .list()
            .layer(
                ConvolutionLayer.Builder(5, 5)
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(1)
                    .activation(Activation.IDENTITY)
                    .build()
            )
            /*.layer(
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
            )*/
            .layer(
                OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build()
            )
            .setInputType(InputType.convolutionalFlat(28, 28, 1))
            .build()
    }

    abstract fun run()

    fun calculateWeightedAverageParams(params: List<Pair<INDArray, Int>>): INDArray {
        val totalWeight = params.map { it.second }.reduce { sum, elem -> sum + elem }.toDouble()
        var arr: INDArray = params[0].first.mul(params[0].second.toDouble() / totalWeight)
        for (i in 1 until params.size) {
            arr = arr.add(params[i].first.mul(params[i].second.toDouble() / totalWeight))
        }
        return arr
    }
}
