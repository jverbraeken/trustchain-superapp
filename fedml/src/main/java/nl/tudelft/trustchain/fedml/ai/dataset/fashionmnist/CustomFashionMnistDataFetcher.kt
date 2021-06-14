package nl.tudelft.trustchain.fedml.ai.dataset.fashionmnist

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.apache.commons.io.FilenameUtils
import org.apache.commons.io.FileUtils
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.deeplearning4j.datasets.base.EmnistFetcher
import org.deeplearning4j.datasets.base.MnistFetcher
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import java.util.stream.IntStream

private val logger = KotlinLogging.logger("CustomMnistDataFetcher")
private const val NUM_EMNIST_TRAINING_EXAMPLES = 124800
private const val NUM_EMNIST_TESTING_EXAMPLES = 20800
private const val NUM_EMNIST_CLASSES = 26
private const val NUM_MNIST_CLASSES = 10
private const val NUM_EMNIST_EXAMPLES_PER_CLASS = 2200
private const val SIZE_IMAGE = 28 * 28
private const val FILENAME_FASHION_MNIST_TRAIN_IMAGES = "train-images-idx3-ubyte"
private const val FILENAME_FASHION_MNIST_TRAIN_LABELS = "train-labels-idx1-ubyte"
private const val FILENAME_FASHION_MNIST_TEST_IMAGES = "t10k-images-idx3-ubyte"
private const val FILENAME_FASHION_MNIST_TEST_LABELS = "t10k-labels-idx1-ubyte"

class CustomFashionMnistDataFetcher(
    val iteratorDistribution: IntArray,
    seed: Long,
    val dataSetType: CustomDataSetType,
    maxTestSamples: Int,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { createTestBatches() }

    @Transient
    private var man: CustomFashionMnistManager
    private var featureData = Array(1) { FloatArray(SIZE_IMAGE) }

    init {
        if (!fashionMnistExists(transfer)) {
//            (if (transfer) EmnistFetcher(EmnistDataSetIterator.Set.LETTERS) else MnistFetcher()).downloadAndUntar()
        }
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, if (transfer) "FASHION_MNIST" else "FASHION_MNIST").absolutePath
        val images: String
        val labels: String
        val numExamples: Int
        if (dataSetType == CustomDataSetType.TRAIN) {
            images = FilenameUtils.concat(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TRAIN_IMAGES else FILENAME_FASHION_MNIST_TRAIN_IMAGES)
            labels = FilenameUtils.concat(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TRAIN_LABELS else FILENAME_FASHION_MNIST_TRAIN_LABELS)
            numExamples = if (transfer) MnistDataFetcher.NUM_EXAMPLES else MnistDataFetcher.NUM_EXAMPLES
        } else {
            images = FilenameUtils.concat(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TEST_IMAGES else FILENAME_FASHION_MNIST_TEST_IMAGES)
            labels = FilenameUtils.concat(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TEST_LABELS else FILENAME_FASHION_MNIST_TEST_LABELS)
            numExamples = if (transfer) MnistDataFetcher.NUM_EXAMPLES_TEST else MnistDataFetcher.NUM_EXAMPLES_TEST
        }
        val createMan = {
            CustomFashionMnistManager(
                images,
                labels,
                numExamples,
                if (transfer) (0 until 10).map { 1000 }.toIntArray() else iteratorDistribution,
                maxTestSamples,
                seed,
                if (dataSetType == CustomDataSetType.FULL_TEST) Behaviors.BENIGN else behavior
            )
        }
        try {
            man = createMan.invoke()
        } catch (e: Exception) {
            try {
                FileUtils.deleteDirectory(File(mnistRoot))
            } catch (e2: Exception) {
                // Ignore
            }
//            MnistFetcher().downloadAndUntar()
            man = createMan.invoke()
        }
        totalExamples = man.getNumSamples()
        numOutcomes = if (transfer) NUM_EMNIST_CLASSES else NUM_MNIST_CLASSES
        cursor = 0
        inputColumns = man.getInputColumns()
        order = IntStream.range(0, totalExamples).toArray()
        reset() //Shuffle order
    }

    private fun fashionMnistExists(transfer: Boolean): Boolean {
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, if (transfer) "FASHION_MNIST" else "FASHION_MNIST").absolutePath
        var f = File(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TRAIN_IMAGES else FILENAME_FASHION_MNIST_TRAIN_IMAGES)
        if (!f.exists()) return false
        f = File(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TRAIN_LABELS else FILENAME_FASHION_MNIST_TRAIN_LABELS)
        if (!f.exists()) return false
        f = File(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TEST_IMAGES else FILENAME_FASHION_MNIST_TEST_IMAGES)
        if (!f.exists()) return false
        f = File(mnistRoot, if (transfer) FILENAME_FASHION_MNIST_TEST_LABELS else FILENAME_FASHION_MNIST_TEST_LABELS)
        return f.exists()
    }

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more; there are no more images" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), numOutcomes.toLong())
        if (featureData.size != numExamples) {
            featureData = Array(numExamples) { FloatArray(SIZE_IMAGE) }
        }
        var actualExamples = 0
        for (i in 0 until numExamples) {
            if (!hasMore()) break
            val (image, label) = man.readEntry(order[cursor])
            featureData[actualExamples] = image
            labels.put(actualExamples, label, 1.0f)
            actualExamples++
            cursor++
        }
        val features = Nd4j.create(
            if (featureData.size == actualExamples) featureData
            else featureData.copyOfRange(0, actualExamples)
        )
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        features.divi(255.0)
        curr = DataSet(features, labels)
        if (dataSetType != CustomDataSetType.TRAIN) {
            /**
             * Non-training iterators are called not by the next() function but by supplying the whole iterator
             * => require a reset before getting next batch, otherwise the iterator keeps iterating
             */
            cursor = totalExamples
        }
    }

    private fun createTestBatches(): Array<DataSet?> {
        val testBatches = man.createTestBatches()
        if (featureData.size != TEST_BATCH_SIZE) {
            featureData = Array(TEST_BATCH_SIZE) { FloatArray(SIZE_IMAGE) }
        }
        val result = arrayListOf<DataSet?>()
        for ((label, batch) in testBatches.withIndex()) {
            result.add(
                if (batch.isEmpty()) null
                else createTestBatch(label, batch)
            )
        }
        return result.toTypedArray()
    }

    private fun createTestBatch(label: Int, batch: Array<FloatArray>): DataSet {
        val numSamplesInBatch = batch.size
        val labels = Nd4j.zeros(DataType.FLOAT, numSamplesInBatch.toLong(), numOutcomes.toLong())
        for ((i, img) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            featureData[i] = img
        }
        val features = Nd4j.create(featureData)
        features.divi(255.0)
        return DataSet(features, labels)
    }

    val labels: List<String>
        get() = man.getLabels()

    companion object {
        const val TEST_BATCH_SIZE = 10
    }
}
