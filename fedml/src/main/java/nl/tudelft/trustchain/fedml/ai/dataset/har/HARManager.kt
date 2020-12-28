package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import java.io.File
import java.io.IOException

class HARManager(
    dataFiles: Array<File>,
    labelsFile: File,
    iteratorDistribution: List<Int>,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledDataArr: Array<Array<String>>
    private val sampledLabelsArr: IntArray

    init {
        fullDataArr.putIfAbsent(dataFiles[0].name, loadData(dataFiles))
        fullLabelsArr.putIfAbsent(labelsFile.name, loadLabels(labelsFile))
        val labelsArr = fullLabelsArr[labelsFile.name]!!
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
        val res = sampleData(fullDataArr[dataFiles[0].name]!!, labelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        sampledDataArr = res.first
        sampledLabelsArr = res.second
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
        iteratorDistribution: List<Int>,
        maxTestSamples: Int,
        seed: Long,
    ): Pair<Array<Array<String>>, IntArray> {
        val dataArr = Array(totalExamples) { arrayListOf<String>() }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
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
            val correspondingDataIndices = sampledLabelsArr.indices
                .filter { i: Int -> sampledLabelsArr[i] == label }
                .take(50)
                .toTypedArray()
            correspondingDataIndices
                .map { i: Int -> sampledDataArr[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readEntryUnsafe(i: Int): Array<String> {
        return sampledDataArr[i]
    }

    fun readLabel(i: Int): Int {
        return sampledLabelsArr[i]
    }

    fun getNumSamples(): Int {
        return sampledDataArr.size
    }

    fun getLabels(): List<String> {
        return sampledLabelsArr
            .distinct()
            .map { i: Int -> i.toString() }
            .toList()
    }

    companion object {
        private var fullDataArr = hashMapOf<String, Array<List<String>>>()
        private var fullLabelsArr = hashMapOf<String, Array<Int>>()
    }
}
