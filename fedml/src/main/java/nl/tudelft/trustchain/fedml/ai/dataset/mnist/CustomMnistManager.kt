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
    labelsFile: String,
    numExamples: Int,
    iteratorDistribution: IntArray,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledImagesArr: Array<ByteArray>
    private val featureData: Array<FloatArray>
    private val sampledLabelsArr: IntArray

    init {
        fullImagesArr.putIfAbsent(imagesFile, loadImages(imagesFile, numExamples))
        fullLabelsArr.putIfAbsent(labelsFile, loadLabels(labelsFile, numExamples))
        val labelsArr = fullLabelsArr[labelsFile]!!
        val labelsArr2 = labelsArr
            .copyOf(labelsArr.size)
            .map { it }
            .toIntArray()
        if (behavior === Behaviors.LABEL_FLIP) {
            labelsArr.indices
                .filter { i: Int -> labelsArr[i] == 1 }
                .forEach { i: Int -> labelsArr2[i] = 2 }
            labelsArr.indices
                .filter { i: Int -> labelsArr[i] == 2 }
                .forEach { i: Int -> labelsArr2[i] = 1 }
        }
        val totalExamples = calculateTotalExamples(iteratorDistribution, maxTestSamples, labelsArr2)
        val res = sampleData(fullImagesArr[imagesFile]!!, labelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        sampledImagesArr = res.first
        sampledLabelsArr = res.second

        featureData = sampledImagesArr.map {
            it.map { byte ->
                (byte.toInt() and 0xFF).toFloat()
            }.toFloatArray()
        }.toTypedArray()
    }

    private fun sampleData(
        tmpImagesArr: Array<ByteArray>,
        tmpLabelsArr: IntArray,
        totalExamples: Int,
        iteratorDistribution: IntArray,
        maxTestSamples: Int,
        seed: Long,
    ): Pair<Array<ByteArray>, IntArray> {
        val imagesArr = Array(totalExamples) { ByteArray(tmpImagesArr[0].size) }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        val random = Random(seed)
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            val matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr)
            matchingImageIndices.shuffle(random)
            for (j in 0 until minOf(matchingImageIndices.size, maxSamplesOfLabel, maxTestSamples)) {
                imagesArr[count] = tmpImagesArr[matchingImageIndices[j]]
                labelsArr[count] = tmpLabelsArr[matchingImageIndices[j]]
                count++
            }
        }
        return Pair(imagesArr, labelsArr)
    }

    fun createTestBatches(): Array<Array<ByteArray>> {
        return (0 until 10).map { label ->
            val correspondingImageIndices = sampledLabelsArr.indices
                .filter { i: Int -> sampledLabelsArr[i] == label }
                .take(20)
                .toTypedArray()
            correspondingImageIndices
                .map { i -> sampledImagesArr[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readImage(i: Int): FloatArray {
        return featureData[i]
    }

    fun readLabel(i: Int): Int {
        return sampledLabelsArr[i]
    }

    fun getNumSamples(): Int {
        return sampledImagesArr.size
    }

    fun getLabels(): List<String> {
        return sampledLabelsArr
            .distinct()
            .map { i: Int -> i.toString() }
    }

    fun getInputColumns(): Int {
        return getImageEntryLength(imagesFile)
    }

    companion object {
        private val imageMapping = hashMapOf<String, Array<ByteArray>>()
        private val labelMapping = hashMapOf<String, IntArray>()
        private val imageEntryLength = hashMapOf<String, Int>()
        private val fullImagesArr = mutableMapOf<String, Array<ByteArray>>()
        private var fullLabelsArr = mutableMapOf<String, IntArray>()

        @Synchronized
        private fun loadImages(filename: String, numExamples: Int): Array<ByteArray> {
            return imageMapping.getOrPut(filename) {
                val file = MnistImageFile(filename, "r")
                imageEntryLength[filename] = file.entryLength
                file.readImagesUnsafe(numExamples)
            }
        }

        @Synchronized
        private fun loadLabels(filename: String, numExamples: Int): IntArray {
            return labelMapping.getOrPut(filename) {
                val file = MnistLabelFile(filename, "r")
                file.readLabels(numExamples)
            }
        }

        @Synchronized
        private fun getImageEntryLength(imagesFile: String): Int {
            return imageEntryLength[imagesFile]!!
        }
    }
}
