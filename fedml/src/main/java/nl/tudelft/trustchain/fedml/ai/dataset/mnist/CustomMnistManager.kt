package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import org.deeplearning4j.datasets.mnist.MnistImageFile
import org.deeplearning4j.datasets.mnist.MnistLabelFile

private val logger = KotlinLogging.logger("CustomMnistManager")

class CustomMnistManager(
    val imagesFile: String,
    labelsFile: String,
    numExamples: Int,
    iteratorDistribution: List<Int>,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledImagesArr: Array<ByteArray>
    private val sampledLabelsArr: IntArray

    init {
        fullImagesArr.putIfAbsent(imagesFile, loadImages(imagesFile, numExamples))
        fullLabelsArr.putIfAbsent(labelsFile, loadLabels(labelsFile, numExamples))
        val labelsArr = fullLabelsArr[labelsFile]!!
        val labelsArr2 = labelsArr
            .copyOf(labelsArr.size)
            .map { it!! }
            .toTypedArray()
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
    }

    private fun sampleData(
        tmpImagesArr: Array<ByteArray>,
        tmpLabelsArr: Array<Int>,
        totalExamples: Int,
        iteratorDistribution: List<Int>,
        maxTestSamples: Int,
        seed: Long,
    ): Pair<Array<ByteArray>, IntArray> {
        val imagesArr = Array(totalExamples) { ByteArray(tmpImagesArr[0].size) }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            val matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr)
            val shuffledMatchingImageIndices = shuffle(matchingImageIndices, seed)
            for (j in 0 until minOf(shuffledMatchingImageIndices.size, maxSamplesOfLabel, maxTestSamples)) {
                imagesArr[count] = tmpImagesArr[shuffledMatchingImageIndices[j]]
                labelsArr[count] = tmpLabelsArr[shuffledMatchingImageIndices[j]]
                count++
            }
        }
        return Pair(imagesArr, labelsArr)
    }

    fun createTestBatches(): Array<Array<ByteArray>> {
        return (0 until 10).map { label ->
            val correspondingImageIndices = sampledLabelsArr.indices
                .filter { i: Int -> sampledLabelsArr[i] == label }
                .take(50)
                .toTypedArray()
            correspondingImageIndices
                .map { i -> sampledImagesArr[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readImageUnsafe(i: Int): ByteArray {
        return sampledImagesArr[i]
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
            .toList()
    }

    fun getInputColumns(): Int {
        return getImageEntryLength(imagesFile)
    }

    companion object {
        private val imageMapping = hashMapOf<String, Array<ByteArray>>()
        private val labelMapping = hashMapOf<String, Array<Int>>()
        private val imageEntryLength = hashMapOf<String, Int>()
        private val fullImagesArr = mutableMapOf<String, Array<ByteArray>>()
        private var fullLabelsArr = mutableMapOf<String, Array<Int>>()

        @Synchronized
        private fun loadImages(filename: String, numExamples: Int): Array<ByteArray> {
            return imageMapping.getOrPut(filename) {
                val file = MnistImageFile(filename, "r")
                imageEntryLength[filename] = file.entryLength
                file.readImagesUnsafe(numExamples)
            }
        }

        @Synchronized
        private fun loadLabels(filename: String, numExamples: Int): Array<Int> {
            return labelMapping.getOrPut(filename) {
                val file = MnistLabelFile(filename, "r")
                file.readLabels(numExamples).toTypedArray()
            }
        }

        @Synchronized
        private fun getImageEntryLength(imagesFile: String): Int {
            return imageEntryLength[imagesFile]!!
        }
    }
}
