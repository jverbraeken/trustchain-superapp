package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import nl.tudelft.trustchain.fedml.ai.dataset.NUM_FULL_TEST_SAMPLES
import java.io.BufferedReader
import java.io.File
import java.io.FileReader
import java.io.IOException

class HARManager(
    dataFiles: Array<File>,
    labelsFile: File,
    iteratorDistribution: List<Int>?,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors
) : DatasetManager() {
    private val dataArr: Array<Array<String>>
    private val labelsArr: IntArray

    init {
        val tmpDataArr = loadData(dataFiles)
        val tmpLabelsArr = loadLabels(labelsFile)
        val tmpLabelsArr2 = tmpLabelsArr
            .copyOf(tmpLabelsArr.size)
            .map { it!! }
            .toTypedArray()
        if (behavior === Behaviors.LABEL_FLIP) {
            tmpLabelsArr.indices.filter { i: Int -> tmpLabelsArr[i] == 1 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 2 }
            tmpLabelsArr.indices.filter { i: Int -> tmpLabelsArr[i] == 2 }
                .forEach { i: Int -> tmpLabelsArr2[i] = 1 }
        }
        val totalExamples = calculateTotalExamples(iteratorDistribution, maxTestSamples, tmpLabelsArr2)
        val res = sampleData(tmpDataArr, tmpLabelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        dataArr = res.first
        labelsArr = res.second
    }

    @Throws(IOException::class)
    private fun loadData(dataFiles: Array<File>): Array<List<String>> {
        return dataFiles.map {
            it.readLines()
        }.toTypedArray()
    }

    @Throws(IOException::class)
    private fun loadLabels(labelsFile: File): Array<Int> {
        return labelsFile
            .readLines()
            .map { i: String -> i.toInt() - 1 } // labels start at 1 instead of 0
            .toTypedArray()
    }

    private fun sampleData(
        tmpDataArr: Array<List<String>>,
        tmpLabelsArr: Array<Int>,
        totalExamples: Int,
        iteratorDistribution: List<Int>?,
        maxTestSamples: Int,
        seed: Long
    ): Pair<Array<Array<String>>, IntArray> {
        val dataArr = Array(totalExamples) { arrayListOf<String>() }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        for (label in iteratorDistribution?.indices ?: tmpLabelsArr.distinct()) {
            val maxSamplesOfLabel = iteratorDistribution?.get(label) ?: NUM_FULL_TEST_SAMPLES
            val matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr)
            val shuffledMatchingImageIndices = shuffle(matchingImageIndices, seed)
            for (j in 0 until minOf(shuffledMatchingImageIndices.size, maxSamplesOfLabel, maxTestSamples)) {
                dataArr[count].addAll(tmpDataArr
                    .map { dataFile -> dataFile[shuffledMatchingImageIndices[j]] }
                    .toList()
                )
                labelsArr[count] = tmpLabelsArr[shuffledMatchingImageIndices[j]]
                count++
            }
        }
        return Pair(dataArr.map { it.toTypedArray() }.toTypedArray(), labelsArr)
    }

    fun createTestBatches(): Array<Array<Array<String>>> {
        return (0 until HARDataFetcher.NUM_LABELS).map { label ->
            val correspondingDataIndices = labelsArr.indices
                .filter { i: Int -> labelsArr[i] == label }
                .take(50)
                .toTypedArray()
            correspondingDataIndices
                .map { i: Int -> dataArr[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readEntryUnsafe(i: Int): Array<String> {
        return dataArr[i]
    }

    fun readLabel(i: Int): Int {
        return labelsArr[i]
    }

    fun getNumSamples(): Int {
        return dataArr.size
    }

    fun getLabels(): List<String> {
        return labelsArr
            .distinct()
            .map { i: Int -> i.toString() }
            .toList()
    }
}
