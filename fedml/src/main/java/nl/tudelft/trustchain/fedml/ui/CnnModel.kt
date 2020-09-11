package nl.tudelft.trustchain.fedml.ui

//import org.deeplearning4j.eval.Evaluation
//import org.deeplearning4j.nn.api.OptimizationAlgorithm
//import org.deeplearning4j.nn.conf.MultiLayerConfiguration
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration
//import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
//import org.deeplearning4j.nn.conf.layers.OutputLayer
//import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
//import org.deeplearning4j.nn.weights.WeightInit
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.lossfunctions.LossFunctions
//
//
//internal class CnnModel(
//    private val dataSetService: CifarDataSetService,
//    private val properties: CnnModelProperties
//) {
//    private val network: MultiLayerNetwork
//
//    init {
//        val configuration: MultiLayerConfiguration = NeuralNetConfiguration.Builder()
//            .seed(1611)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .updater(properties.optimizer)
//            .list()
//            .layer(0, conv5x5())
//            .layer(1, pooling2x2Stride2())
//            .layer(2, conv3x3Stride1Padding2())
//            .layer(3, pooling2x2Stride1())
//            .layer(4, conv3x3Stride1Padding1())
//            .layer(5, pooling2x2Stride1())
//            .layer(6, dense())
//            .setInputType(dataSetService.inputType)
//            .build()
//        network = MultiLayerNetwork(configuration)
//    }
//
//    fun train() {
//        network.init()
//        val epochsNum: Int = properties.epochsNum
//        for (i in 1..epochsNum) {
////            log.info("Epoch {} / {}", epoch, epochsNum)
////            network.fit(dataSetService.trainIterator)
//        }
//    }
//
////    fun evaluate(): Evaluation {
////        return network.evaluate(dataSetService.testIterator)
////    }
//
//    private fun conv5x5(): ConvolutionLayer {
//        return ConvolutionLayer.Builder(5, 5)
//            .nIn(3)
//            .nOut(16)
//            .stride(1, 1)
//            .padding(1, 1)
//            .weightInit(WeightInit.XAVIER_UNIFORM)
//            .activation(Activation.RELU)
//            .build()
//    }
//
//    private fun pooling2x2Stride2(): SubsamplingLayer {
//        return SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//            .kernelSize(2, 2)
//            .stride(2, 2)
//            .build()
//    }
//
//    private fun conv3x3Stride1Padding2(): ConvolutionLayer {
//        return ConvolutionLayer.Builder(3, 3)
//            .nOut(32)
//            .stride(1, 1)
//            .padding(2, 2)
//            .weightInit(WeightInit.XAVIER_UNIFORM)
//            .activation(Activation.RELU)
//            .build()
//    }
//
//    private fun pooling2x2Stride1(): SubsamplingLayer {
//        return SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
//            .kernelSize(2, 2)
//            .stride(1, 1)
//            .build()
//    }
//
//    private fun conv3x3Stride1Padding1(): ConvolutionLayer {
//        return ConvolutionLayer.Builder(3, 3)
//            .nOut(64)
//            .stride(1, 1)
//            .padding(1, 1)
//            .weightInit(WeightInit.XAVIER_UNIFORM)
//            .activation(Activation.RELU)
//            .build()
//    }
//
//    private fun dense(): OutputLayer {
//        return OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
//            .activation(Activation.SOFTMAX)
//            .weightInit(WeightInit.XAVIER_UNIFORM)
//            .nOut(dataSetService.labels().size - 1)
//            .build()
//    }
//}
