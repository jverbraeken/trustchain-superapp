package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import org.deeplearning4j.datasets.mnist.MnistImageFile
import org.deeplearning4j.datasets.mnist.MnistLabelFile
import java.io.IOException
import java.util.*
import java.util.stream.Collectors
import java.util.stream.IntStream

class CustomMnistManager(
    imagesFile: String?,
    labelsFile: String?,
    numExamples: Int,
    iteratorDistribution: List<Int>,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors
) : DatasetManager() {
    private val mnistImageFile = MnistImageFile(imagesFile, "r")
    private val mnistLabelFile = MnistLabelFile(labelsFile, "r")
    private val imagesArr: Array<ByteArray>
    private val labelsArr: IntArray

    init {
        val tmpImagesArr = loadImages(mnistImageFile, numExamples)
        val tmpLabelsArr = loadLabels(mnistLabelFile, numExamples)
        val tmpLabelsArr2 = tmpLabelsArr.copyOf(tmpLabelsArr.size)
        if (behavior === Behaviors.LABEL_FLIP) {
            IntStream.range(0, tmpLabelsArr.size).filter { i: Int -> tmpLabelsArr[i] == 1 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 2 }
            IntStream.range(0, tmpLabelsArr.size).filter { i: Int -> tmpLabelsArr[i] == 2 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 1 }
        }
        val totalExamples = calculateTotalExamples(iteratorDistribution, maxTestSamples, tmpLabelsArr2)
        val res = sampleData(tmpImagesArr, tmpLabelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        imagesArr = res.first
        labelsArr = res.second
    }

    @Throws(IOException::class)
    private fun loadLabels(mnistLabelFile: MnistLabelFile, numExamples: Int): IntArray {
        return mnistLabelFile.readLabels(numExamples)
    }

    @Throws(IOException::class)
    private fun loadImages(mnistImageFile: MnistImageFile, numExamples: Int): Array<ByteArray> {
        return mnistImageFile.readImagesUnsafe(numExamples)
    }

    private fun sampleData(
        tmpImagesArr: Array<ByteArray>,
        tmpLabelsArr: IntArray,
        totalExamples: Int,
        iteratorDistribution: List<Int>,
        maxTestSamples: Int,
        seed: Long
    ): Pair<Array<ByteArray>, IntArray> {
        val imagesArr = Array(totalExamples) { ByteArray(tmpImagesArr[0].size) }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            val matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr)
            val shuffledMatchingImageIndices = shuffle(matchingImageIndices, seed)
            for (j in 0 until min(shuffledMatchingImageIndices.size, maxSamplesOfLabel, maxTestSamples)) {
                imagesArr[count] = tmpImagesArr[shuffledMatchingImageIndices[j]]
                labelsArr[count] = tmpLabelsArr[shuffledMatchingImageIndices[j]]
                count++
            }
        }
        return Pair(imagesArr, labelsArr)
    }

    fun createTestBatches(): Array<Array<ByteArray>> {
        return (0 until 10).map { label ->
            val correspondingImageIndices = IntStream
                .range(0, labelsArr.size)
                .filter { i: Int -> labelsArr[i] == label }
                .limit(50)
                .toArray()
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

    /**
     * Get the underlying images file as [MnistImageFile].
     *
     * @return [MnistImageFile].
     */
    fun getImages(): MnistImageFile {
        return mnistImageFile
    }

    fun getLabels(): List<String> {
        return Arrays.stream(labelsArr).distinct().mapToObj { i: Int -> i.toString() }.collect(Collectors.toList())
    }
}
