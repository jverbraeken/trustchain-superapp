package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import nl.tudelft.trustchain.fedml.ai.dataset.NUM_FULL_TEST_SAMPLES
import org.deeplearning4j.datasets.mnist.MnistImageFile
import org.deeplearning4j.datasets.mnist.MnistLabelFile

private val logger = KotlinLogging.logger("CustomMnistManager")

class CustomMnistManager(
    val imagesFile: String,
    labelsFile: String,
    numExamples: Int,
    iteratorDistribution: List<Int>?,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors
) : DatasetManager() {
    private val imagesArr: Array<ByteArray>
    private val labelsArr: IntArray

    init {
        val tmpImagesArr = loadImages(imagesFile, numExamples)
        val tmpLabelsArr = loadLabels(labelsFile, numExamples)
        val tmpLabelsArr2 = tmpLabelsArr
            .copyOf(tmpLabelsArr.size)
            .map {it!!}
            .toTypedArray()
        if (behavior === Behaviors.LABEL_FLIP) {
            tmpLabelsArr.indices
                .filter { i: Int -> tmpLabelsArr[i] == 1 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 2 }
            tmpLabelsArr.indices
                .filter { i: Int -> tmpLabelsArr[i] == 2 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 1 }
        }
        val totalExamples = calculateTotalExamples(iteratorDistribution, maxTestSamples, tmpLabelsArr2)
        val res = sampleData(tmpImagesArr, tmpLabelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        imagesArr = res.first
        labelsArr = res.second
    }

    private fun sampleData(
        tmpImagesArr: Array<ByteArray>,
        tmpLabelsArr: Array<Int>,
        totalExamples: Int,
        iteratorDistribution: List<Int>?,
        maxTestSamples: Int,
        seed: Long
    ): Pair<Array<ByteArray>, IntArray> {
        val imagesArr = Array(totalExamples) { ByteArray(tmpImagesArr[0].size) }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        for (label in iteratorDistribution?.indices ?: tmpLabelsArr.distinct()) {
            val maxSamplesOfLabel = iteratorDistribution?.get(label) ?: NUM_FULL_TEST_SAMPLES
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
            val correspondingImageIndices = labelsArr.indices
                .filter { i: Int -> labelsArr[i] == label }
                .take(50)
                .toTypedArray()
            correspondingImageIndices
                .map { i -> imagesArr[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readImageUnsafe(i: Int): ByteArray {
        return imagesArr[i]
    }

    fun readLabel(i: Int): Int {
        return labelsArr[i]
    }

    fun getNumSamples(): Int {
        return imagesArr.size
    }

    fun getLabels(): List<String> {
        return labelsArr
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

        @Synchronized private fun loadImages(filename: String, numExamples: Int): Array<ByteArray> {
            return imageMapping.getOrPut(filename) {
                val file = MnistImageFile(filename, "r")
                imageEntryLength[filename] = file.entryLength
                file.readImagesUnsafe(numExamples)
            }
        }

        @Synchronized private fun loadLabels(filename: String, numExamples: Int): Array<Int> {
            return labelMapping.getOrPut(filename) {
                val file = MnistLabelFile(filename, "r")
                file.readLabels(numExamples).toTypedArray()
            }
        }

        @Synchronized private fun getImageEntryLength(imagesFile: String): Int {
            return imageEntryLength[imagesFile]!!
        }
    }
}
