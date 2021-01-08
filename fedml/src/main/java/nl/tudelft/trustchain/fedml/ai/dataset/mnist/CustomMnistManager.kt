package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import org.deeplearning4j.datasets.mnist.MnistImageFile
import org.deeplearning4j.datasets.mnist.MnistLabelFile
import kotlin.random.Random

private val logger = KotlinLogging.logger("CustomMnistManager")

class CustomMnistManager(
    val imagesFile: String,
    val labelsFile: String,
    numExamples: Int,
    iteratorDistribution: IntArray,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledImages: Array<FloatArray>
    private val sampledLabels: IntArray

    init {
        fullImagesArr.computeIfAbsent(imagesFile) { loadImages(imagesFile, numExamples) }
        fullLabelsArr.computeIfAbsent(labelsFile) { loadLabels(labelsFile, numExamples) }
        labelIndexMappings.computeIfAbsent(labelsFile) { generateLabelIndexMapping(fullLabelsArr[labelsFile]!!) }
        val imagesArr = fullImagesArr[imagesFile]!!
        val labelIndexMapping = labelIndexMappings[labelsFile]!!

        val labelIndexMapping2 = if (behavior === Behaviors.LABEL_FLIP) {
            val labelIndexMapping2 = labelIndexMapping.map { it.copyOf() }.toTypedArray()
            labelIndexMapping2[1] = labelIndexMapping[2]
            labelIndexMapping2[2] = labelIndexMapping[3]
            labelIndexMapping2[3] = labelIndexMapping[1]
            labelIndexMapping2
        } else {
            labelIndexMapping
        }
        val totalExamples = calculateTotalExamples(labelIndexMapping2, iteratorDistribution, maxTestSamples)
        val res = sampleData(imagesArr, totalExamples, iteratorDistribution, maxTestSamples, seed, labelIndexMapping2)
        sampledImages = res.first
        sampledLabels = res.second
    }

    private fun sampleData(
        tmpImagesArr: Array<ByteArray>,
        totalExamples: Int,
        iteratorDistribution: IntArray,
        maxTestSamples: Int,
        seed: Long,
        labelIndexMapping: Array<IntArray>,
    ): Pair<Array<FloatArray>, IntArray> {
        val placeholder = Pair(byteArrayOf(), -1)
        val results = Array(totalExamples) { placeholder }
        var count = 0
        val random = Random(seed)
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            labelIndexMapping[label].shuffle(random)
            for (j in 0 until minOf(labelIndexMapping[label].size, maxSamplesOfLabel, maxTestSamples)) {
                results[count] = Pair(tmpImagesArr[labelIndexMapping[label][j]], label)
                count++
            }
        }
        results.shuffle(random)
        val imagesArr = Array(totalExamples) { byteArrayOf() }
        val labelsArr = IntArray(totalExamples)
        results.forEachIndexed { i, pair ->
            imagesArr[i] = pair.first
            labelsArr[i] = pair.second
        }
        val imagesArr2 = imagesArr.map {
            it.map { byte ->
                (byte.toInt() and 0xFF).toFloat()
            }.toFloatArray()
        }.toTypedArray()
        return Pair(imagesArr2, labelsArr)
    }

    fun createTestBatches(): Array<Array<FloatArray>> {
        return (0 until 10).map { label ->
            val correspondingImageIndices = sampledLabels.indices
                .filter { i: Int -> sampledLabels[i] == label }
                .take(CustomMnistDataFetcher.TEST_BATCH_SIZE)
                .toTypedArray()
            correspondingImageIndices
                .map { i -> sampledImages[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readEntry(i: Int): Pair<FloatArray, Int> {
        return Pair(sampledImages[i], sampledLabels[i])
    }

    fun getNumSamples(): Int {
        return sampledImages.size
    }

    fun getLabels(): List<String> {
        return sampledLabels
            .distinct()
            .map { i: Int -> i.toString() }
    }

    fun getInputColumns(): Int {
        return getImageEntryLength(imagesFile)
    }

    companion object {
        private val imageEntryLength = hashMapOf<String, Int>()
        private val fullImagesArr = mutableMapOf<String, Array<ByteArray>>()
        private var fullLabelsArr = mutableMapOf<String, IntArray>()
        private var labelIndexMappings = mutableMapOf<String, Array<IntArray>>()

        @Synchronized
        private fun loadImages(filename: String, numExamples: Int): Array<ByteArray> {
            val file = MnistImageFile(filename, "r")
            imageEntryLength[filename] = file.entryLength
            return file.readImagesUnsafe(numExamples)
        }

        @Synchronized
        private fun loadLabels(filename: String, numExamples: Int): IntArray {
            val file = MnistLabelFile(filename, "r")
            return file.readLabels(numExamples)
        }

        @Synchronized
        private fun generateLabelIndexMapping(labelsArr: IntArray): Array<IntArray> {
            return (0 until CustomMnistDataFetcher.NUM_LABELS).map { label ->
                labelsArr.indices.filter { labelsArr[it] == label }.toIntArray()
            }.toTypedArray()
        }

        @Synchronized
        private fun getImageEntryLength(imagesFile: String): Int {
            return imageEntryLength[imagesFile]!!
        }
    }
}
