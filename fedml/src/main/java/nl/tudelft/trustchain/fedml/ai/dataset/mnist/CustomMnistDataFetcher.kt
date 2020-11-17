package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.apache.commons.io.FileUtils
import org.apache.commons.io.FilenameUtils
import org.deeplearning4j.base.MnistFetcher
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.util.MathUtils
import java.io.File
import java.util.*
import java.util.stream.IntStream
import java.util.zip.Adler32
import java.util.zip.Checksum

const val CHECKSUM_TRAIN_FEATURES = 2094436111L
const val CHECKSUM_TRAIN_LABELS = 4008842612L
const val CHECKSUM_TEST_FEATURES = 2165396896L
const val CHECKSUM_TEST_LABELS = 2212998611L
val CHECKSUMS_TRAIN = longArrayOf(CHECKSUM_TRAIN_FEATURES, CHECKSUM_TRAIN_LABELS)
val CHECKSUMS_TEST = longArrayOf(CHECKSUM_TEST_FEATURES, CHECKSUM_TEST_LABELS)

class CustomMnistDataFetcher(
    iteratorDistribution: List<Int>,
    seed: Long,
    dataSetType: DataSetType,
    maxTestSamples: Int,
    behavior: Behaviors
) : CustomBaseDataFetcher() {
    override val testBatches: List<DataSet> by lazy { createTestBatches() }

    @Transient
    private var man: CustomMnistManager
    private var order: IntArray
    private var rng: Random
    private lateinit var featureData: Array<FloatArray>

    init {
        if (!mnistExists()) {
            MnistFetcher().downloadAndUntar()
        }
        val mnistRoot = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").absolutePath
        val images: String
        val labels: String
        val checksums: LongArray
        val maxExamples: Int
        if (dataSetType == DataSetType.TRAIN) {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TRAINING_FILE_LABELS_FILENAME_UNZIPPED)
            maxExamples = MnistDataFetcher.NUM_EXAMPLES
            checksums = CHECKSUMS_TRAIN
            man = CustomMnistManager(images, labels, maxExamples, iteratorDistribution, Int.MAX_VALUE, seed, behavior)
        } else {
            images = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILES_FILENAME_UNZIPPED)
            labels = FilenameUtils.concat(mnistRoot, MnistFetcher.TEST_FILE_LABELS_FILENAME_UNZIPPED)
            maxExamples = MnistDataFetcher.NUM_EXAMPLES_TEST
            checksums = CHECKSUMS_TEST
            man = CustomMnistManager(images, labels, maxExamples, iteratorDistribution, maxTestSamples, seed, behavior)
        }
        val files = arrayOf(images, labels)
        try {
            man = CustomMnistManager(
                images,
                labels,
                maxExamples,
                iteratorDistribution,
                if (dataSetType == DataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
                seed,
                behavior
            )
            validateFiles(files, checksums)
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
                if (dataSetType == DataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
                seed,
                behavior
            )
            validateFiles(files, checksums)
        }
        totalExamples = man.getNumSamples()
        numOutcomes = 10
        cursor = 0
        inputColumns = man.getImages().entryLength
        order = IntStream.range(0, totalExamples).toArray()
        rng = Random(seed)
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

    private fun validateFiles(files: Array<String>, checksums: LongArray) {
        try {
            for (i in files.indices) {
                val f = File(files[i])
                val adler: Checksum = Adler32()
                val checksum = if (f.exists()) FileUtils.checksum(f, adler).value else -1
                check(!(!f.exists() || checksum != checksums[i])) {
                    "Failed checksum: expected " + checksums[i] +
                        ", got " + checksum + " for file: " + f
                }
            }
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
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
        val features: INDArray
        features = if (featureData.size == actualExamples) {
            Nd4j.create(featureData)
        } else {
            Nd4j.create(featureData.copyOfRange(0, actualExamples))
        }
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        features.divi(255.0)
        curr = DataSet(features, labels)
    }

    private fun createTestBatches(): List<DataSet> {
        val testBatches = man.createTestBatches()
        if (featureData.size < testBatches[0].size) {
            featureData = Array(testBatches[0].size) { FloatArray(28 * 28) }
        }
        var actualExamples = 0
        val result = arrayListOf<DataSet>()
        for ((label, batch) in testBatches.withIndex()) {
            val labels = Nd4j.zeros(DataType.FLOAT, testBatches[0].size.toLong(), numOutcomes.toLong())
            for (img in batch) {
                labels.put(actualExamples, label, 1.0f)
                for (j in img.indices) {
                    featureData[actualExamples][j] = (img[j].toInt() and 0xFF).toFloat()
                }
                actualExamples++
            }
            val features = Nd4j.create(featureData.copyOf())
            features.divi(255.0)
            result.add(DataSet(features, labels))
        }
        return result
    }

    override fun reset() {
        cursor = 0
        curr = null
        MathUtils.shuffleArray(order, rng)
    }

    val labels: List<String>
        get() = man.getLabels()
}
