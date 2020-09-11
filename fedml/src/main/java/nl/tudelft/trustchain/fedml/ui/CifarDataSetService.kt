package nl.tudelft.trustchain.fedml.ui

//import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
//import org.deeplearning4j.nn.conf.inputs.InputType
//
//
//class CifarDataSetService {
//    val inputType: InputType = InputType.convolutional(32, 32, 3)
//    val trainIterator: CifarDataSetIterator
//    val testIterator: CifarDataSetIterator
//    private val trainImagesNum = 512
//    private val testImagesNum = 128
//    private val trainBatch = 16
//    private val testBatch = 8
//
//    fun labels(): List<String> {
//        return trainIterator.labels
//    }
//
//    init {
//        trainIterator = CifarDataSetIterator(trainBatch, trainImagesNum, true)
//        testIterator = CifarDataSetIterator(testBatch, testImagesNum, false)
//    }
//}
