package nl.tudelft.trustchain.fedml.ai.dataset.har_unused

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import java.io.File
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.roundToInt
import kotlin.random.Random

private val logger = KotlinLogging.logger("HARManager")

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
        fullDataArr.computeIfAbsent(dataFiles[0].name) { loadData(dataFiles) }
        fullLabelsArr.computeIfAbsent(labelsFile.name) { loadLabels(labelsFile) }
        labelIndexMappings.computeIfAbsent(labelsFile.name) { generateLabelIndexMapping(fullLabelsArr[labelsFile.name]!!) }
        val dataArr = fullDataArr[dataFiles[0].name]!!
        val labelIndexMapping = labelIndexMappings[labelsFile.name]!!

        val labelIndexMapping2 = if (behavior == Behaviors.LABEL_FLIP_2) {
            val labelIndexMapping2 = labelIndexMapping.map { it.copyOf() }.toTypedArray()
            labelIndexMapping2[0] = labelIndexMapping[1]
            labelIndexMapping2[1] = labelIndexMapping[0]
            labelIndexMapping2
        } else if (behavior == Behaviors.LABEL_FLIP_ALL) {
            val labelIndexMapping2 = labelIndexMapping.map { it.copyOf() }.toTypedArray()
            labelIndexMapping2[0] = labelIndexMapping[1]
            labelIndexMapping2[1] = labelIndexMapping[2]
            labelIndexMapping2[2] = labelIndexMapping[3]
            labelIndexMapping2[3] = labelIndexMapping[4]
            labelIndexMapping2[4] = labelIndexMapping[5]
            labelIndexMapping2[5] = labelIndexMapping[0]
            labelIndexMapping2
        } else {
            labelIndexMapping
        }
        val totalExamples = calculateTotalExamples(labelIndexMapping, iteratorDistribution, maxTestSamples)
        val res = sampleData(dataArr, totalExamples, iteratorDistribution, maxTestSamples, seed, labelIndexMapping2)
        sampledDataArr = res.first
        sampledLabelsArr = res.second
        featureData = sampledDataArr.map { extractFeatures(it) }.toTypedArray()
    }

    private fun extractFeatures(entries: Array<String>): Array<FloatArray> {
        val features = entries.map {
            val trimmed = it.trim()
            val parts = trimmed.split(REGEX_SPLIT).toTypedArray()
            parts.map { s: String -> s.toFloat() }.toTypedArray()
        }.toTypedArray()
        val transposed = transposeMatrix(features)
        return (0 until 50).map { i ->
            if (i % 2 == 0) transposed[(i * 2.5).roundToInt()]
            else transposed[floor(i * 2.5).toInt()].zip(transposed[ceil(i * 2.5).toInt()]).map { (e1, e2) -> (e1 + e2) / 2 }.toFloatArray()
        }.toTypedArray()
    }

    private fun sampleData(
        tmpDataArr: Array<Array<String>>,
        totalExamples: Int,
        iteratorDistribution: IntArray,
        maxTestSamples: Int,
        seed: Long,
        labelIndexMapping: Array<IntArray>,
    ): Pair<Array<Array<String>>, IntArray> {
        val placeholder = Pair<List<String>, Int>(listOf(), -1)
        val results = Array(totalExamples) { placeholder }
        var count = 0
        val random = Random(seed)
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            labelIndexMapping[label].shuffle(random)
            for (j in 0 until minOf(labelIndexMapping[label].size, maxSamplesOfLabel, maxTestSamples)) {
                results[count] = Pair(
                    tmpDataArr.map { dataFile -> dataFile[labelIndexMapping[label][j]] },
                    label
                )
                count++
            }
        }
        results.shuffle(random)
        val dataArr = Array<List<String>>(totalExamples) { listOf() }
        val labelsArr = IntArray(totalExamples)
        results.forEachIndexed { i, pair ->
            dataArr[i] = pair.first
            labelsArr[i] = pair.second
        }
        return Pair(dataArr.map { it.toTypedArray() }.toTypedArray(), labelsArr)
    }

    fun createTestBatches(): Array<Array<Array<FloatArray>>> {
        return (0 until HARDataFetcher.NUM_LABELS).map { label ->
            val correspondingDataIndices = sampledLabelsArr.indices
                .filter { i -> sampledLabelsArr[i] == label }
                .take(HARDataFetcher.TEST_BATCH_SIZE)
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
            .sorted()
            .map { it.toString() }
    }

    companion object {
        private var fullDataArr = hashMapOf<String, Array<Array<String>>>()
        private var fullLabelsArr = hashMapOf<String, IntArray>()
        private var labelIndexMappings = mutableMapOf<String, Array<IntArray>>()
        private val REGEX_SPLIT = "\\s+".toRegex()

        @Synchronized
        private fun loadData(dataFiles: Array<File>): Array<Array<String>> {
            return dataFiles.map {
                it.bufferedReader().readLines().toTypedArray()
            }.toTypedArray()
        }

        @Synchronized
        private fun loadLabels(labelsFile: File): IntArray {
            return labelsFile
                .bufferedReader()
                .readLines()
                .map { s -> s.toInt() - 1 }  // labels start at 1 instead of 0
                .toIntArray()
        }

        @Synchronized
        private fun generateLabelIndexMapping(labelsArr: IntArray): Array<IntArray> {
            return (0 until HARDataFetcher.NUM_LABELS).map { label ->
                labelsArr.indices.filter { labelsArr[it] == label }.toIntArray()
            }.toTypedArray()
        }
    }
}
