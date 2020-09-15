package nl.tudelft.trustchain.fedml.ui

import java.io.InputStream


class CifarDataSetService {
//    val inputType: InputType = InputType.convolutional(32, 32, 3)
//    val trainIterator: Cifar10DataSetIterator = Cifar10DataSetIterator(16)
//    val testIterator: Cifar10DataSetIterator = Cifar10DataSetIterator(8)
//    private val trainImagesNum = 512
//    private val testImagesNum = 128
//    private val trainBatch = 16
//    private val testBatch = 8
//
//    fun labels(): List<String> {
//        return ArrayList()
//        return trainIterator.labels
//    }

    init {
        val file = "res/mnist_png/training"
        val stream: InputStream = this.javaClass.classLoader?.getResourceAsStream(file)!!
        print(stream)
    }

    fun c(): Unit {

    }
}
