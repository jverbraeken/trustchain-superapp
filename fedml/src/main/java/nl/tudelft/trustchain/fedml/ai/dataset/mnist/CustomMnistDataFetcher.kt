package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.base.MnistFetcher
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import java.util.*
import java.util.stream.IntStream

class CustomMnistDataFetcher(
    iteratorDistribution: List<Int>,
    seed: Long,
    dataSetType: CustomDataSetType,
    maxTestSamples: Int,
    behavior: Behaviors,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { createTestBatches() }

    @Transient
    private var man: CustomMnistManager
    private var featureData = Array(1) { FloatArray(28 * 28) }

    init {
        if (!mnistExists()) {
            MnistFetcher().downloadAndUntar()
        }
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").absolutePath
        val images: String
        val labels: String
        val maxExamples: Int
        if (dataSetType == CustomDataSetType.TRAIN) {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED)
            maxExamples = MnistDataFetcher.NUM_EXAMPLES
        } else {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED)
            maxExamples = MnistDataFetcher.NUM_EXAMPLES_TEST
        }
        try {
            man = CustomMnistManager(
                images,
                labels,
                maxExamples,
                iteratorDistribution,
                if (dataSetType == CustomDataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
                seed,
                behavior
            )
        } catch (e: Exception) {
            try {
                FileUtils.deleteDirectory(File(mnistRoot))
            } catch (e2: Exception) {
                // Ignore
            }
            MnistFetcher().downloadAndUntar()
            man = CustomMnistManager(
                images,
                labels,
                maxExamples,
                iteratorDistribution,
                if (dataSetType == CustomDataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
                seed,
                behavior
            )
        }
        totalExamples = man.getNumSamples()
        numOutcomes = NUM_LABELS
        cursor = 0
        inputColumns = man.getInputColumns()
        order = IntStream.range(0, totalExamples).toArray()
        reset() //Shuffle order
    }

    private fun mnistExists(): Boolean {
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").absolutePath
        var f = File(mnistRoot, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED)
        if (!f.exists()) return false
        f = File(mnistRoot, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED)
        return f.exists()
    }

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more; there are no more images" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), numOutcomes.toLong())
        if (featureData.size < numExamples) {
            featureData = Array(numExamples) { FloatArray(28 * 28) }
        }
        var actualExamples = 0
        for (i in 0 until numExamples) {
            if (!hasMore()) break
            val img = man.readImageUnsafe(order[cursor])
            val label = man.readLabel(order[cursor])
            labels.put(actualExamples, label, 1.0f)
            for (j in img.indices) {
                featureData[actualExamples][j] = (img[j].toInt() and 0xFF).toFloat()
            }
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
    }

    private fun createTestBatches(): List<DataSet?> {
        val testBatches = man.createTestBatches()
        if (featureData.size < testBatches[0].size) {
            featureData = Array(testBatches[0].size) { FloatArray(28 * 28) }
        }
        val result = arrayListOf<DataSet?>()
        for ((label, batch) in testBatches.withIndex()) {
            result.add(
                if (batch.isEmpty()) null
                else createTestBatch(label, batch)
            )
        }
        return result
    }

    private fun createTestBatch(label: Int, batch: Array<ByteArray>): DataSet {
        val numSamplesInBatch = batch.size
        val labels = Nd4j.zeros(DataType.FLOAT, numSamplesInBatch.toLong(), numOutcomes.toLong())
        for ((i, img) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            for (j in img.indices) {
                featureData[i][j] = (img[j].toInt() and 0xFF).toFloat()
            }
        }
        val features = Nd4j.create(
            if (featureData.size == numSamplesInBatch) featureData
            else featureData.copyOfRange(0, numSamplesInBatch)
        )
        features.divi(255.0)
        return DataSet(features, labels)
    }

    val labels: List<String>
        get() = man.getLabels()

    companion object {
        const val NUM_LABELS = 10
    }
}
