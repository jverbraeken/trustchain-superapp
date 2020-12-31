package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import java.io.File
import java.io.IOException
import kotlin.random.Random

class HARManager(
    dataFiles: Array<File>,
    labelsFile: File,
    iteratorDistribution: IntArray,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledDataArr: Array<Array<String>>
    private val featureData: Array<Array<FloatArray>>
    private val sampledLabelsArr: IntArray

    init {
        fullDataArr.putIfAbsent(dataFiles[0].name, loadData(dataFiles))
        fullLabelsArr.putIfAbsent(labelsFile.name, loadLabels(labelsFile))
        val labelsArr = fullLabelsArr[labelsFile.name]!!
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
        val res = sampleData(fullDataArr[dataFiles[0].name]!!, labelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed)
        sampledDataArr = res.first
        sampledLabelsArr = res.second

        featureData = sampledDataArr.map { extractFeatures(it) }.toTypedArray()
    }

    private fun extractFeatures(entries: Array<String>): Array<FloatArray> {
        val features = entries.map {
            val trimmed = it.trim()
            val parts = trimmed.split("\\s+".toRegex()).toTypedArray()
            parts.map { s: String -> s.toFloat() }.toTypedArray()
        }.toTypedArray()
        return transposeMatrix(features)
    }

    @Throws(IOException::class)
    private fun loadData(dataFiles: Array<File>): Array<Array<String>> {
        return dataFiles.map {
            it.bufferedReader().readLines().toTypedArray()
        }.toTypedArray()
    }

    @Throws(IOException::class)
    private fun loadLabels(labelsFile: File): IntArray {
        return labelsFile
            .bufferedReader()
            .readLines()
            .map { s -> s.toInt() - 1 }  // labels start at 1 instead of 0
            .toIntArray()
    }

    private fun sampleData(
        tmpDataArr: Array<Array<String>>,
        tmpLabelsArr: IntArray,
        totalExamples: Int,
        iteratorDistribution: IntArray,
        maxTestSamples: Int,
        seed: Long,
    ): Pair<Array<Array<String>>, IntArray> {
        val dataArr = Array(totalExamples) { arrayListOf<String>() }
        val labelsArr = IntArray(totalExamples)
        var count = 0
        val random = Random(seed)
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            val matchingImageIndices = findMatchingImageIndices(label, tmpLabelsArr)
            matchingImageIndices.shuffle(random)
            for (j in 0 until minOf(matchingImageIndices.size, maxSamplesOfLabel, maxTestSamples)) {
                dataArr[count].addAll(tmpDataArr
                    .map { dataFile -> dataFile[matchingImageIndices[j]] }
                )
                labelsArr[count] = tmpLabelsArr[matchingImageIndices[j]]
                count++
            }
        }
        return Pair(dataArr.map { it.toTypedArray() }.toTypedArray(), labelsArr)
    }

    fun createTestBatches(): Array<Array<Array<FloatArray>>> {
        return (0 until HARDataFetcher.NUM_LABELS).map { label ->
            val correspondingDataIndices = sampledLabelsArr.indices
                .filter { i: Int -> sampledLabelsArr[i] == label }
                .take(20)
                .toTypedArray()
            correspondingDataIndices
                .map { i: Int -> featureData[i] }
                .toTypedArray()
        }.toTypedArray()
    }

    fun readEntry(i: Int): Array<FloatArray> {
        return featureData[i]
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
    }

    companion object {
        private var fullDataArr = hashMapOf<String, Array<Array<String>>>()
        private var fullLabelsArr = hashMapOf<String, IntArray>()
    }
}
