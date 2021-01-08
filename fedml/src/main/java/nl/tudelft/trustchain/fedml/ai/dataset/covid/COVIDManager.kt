package nl.tudelft.trustchain.fedml.ai.dataset.covid

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.DatasetManager
import nl.tudelft.trustchain.fedml.ai.dataset.cifar.CustomCifar10Fetcher
import nl.tudelft.trustchain.fedml.ai.dataset.har.transposeMatrix
import org.nd4j.linalg.api.rng.distribution.BaseDistribution
import java.io.File
import java.util.*
import kotlin.random.Random

private val logger = KotlinLogging.logger("HARManager")

class COVIDManager(
    baseDirectory: File,
    iteratorDistribution: IntArray,
    maxTestSamples: Int,
    seed: Long,
    behavior: Behaviors,
) : DatasetManager() {
    private val sampledDataArr: Array<FloatArray>
    private val sampledLabelsArr: IntArray

    init {
        if (fullDataArr == null) {
            val res = loadData(baseDirectory)
            fullDataArr = res.first
            fullLabelsArr = res.second
            labelIndexMapping = generateLabelIndexMapping(fullLabelsArr!!)
        }

        val labelsArr2 = if (behavior === Behaviors.LABEL_FLIP) {
            val labelsArr2 = fullLabelsArr!!.copyOf()
            fullLabelsArr!!.indices
                .filter { i: Int -> fullLabelsArr!![i] == 1 }
                .forEach { i: Int -> labelsArr2[i] = 2 }
            fullLabelsArr!!.indices
                .filter { i: Int -> fullLabelsArr!![i] == 2 }
                .forEach { i: Int -> labelsArr2[i] = 3 }
            fullLabelsArr!!.indices
                .filter { i: Int -> fullLabelsArr!![i] == 3 }
                .forEach { i: Int -> labelsArr2[i] = 1 }
            labelsArr2
        } else {
            fullLabelsArr!!
        }
        val totalExamples = calculateTotalExamples(labelIndexMapping!!, iteratorDistribution, maxTestSamples)
        val res = sampleData(fullDataArr!!, labelsArr2, totalExamples, iteratorDistribution, maxTestSamples, seed, labelIndexMapping!!)
        sampledDataArr = res.first
        sampledLabelsArr = res.second
    }

    private fun sampleData(
        tmpDataArr: Array<FloatArray>,
        tmpLabelsArr: IntArray,
        totalExamples: Int,
        iteratorDistribution: IntArray,
        maxTestSamples: Int,
        seed: Long,
        labelIndexMapping: Array<IntArray>,
    ): Pair<Array<FloatArray>, IntArray> {
        val placeholder = Pair(floatArrayOf(), -1)
        val results = Array(totalExamples) { placeholder }
        var count = 0
        val random = Random(seed)
        for (label in iteratorDistribution.indices) {
            val maxSamplesOfLabel = iteratorDistribution[label]
            labelIndexMapping[label].shuffle(random)
            for (j in 0 until minOf(labelIndexMapping[label].size, maxSamplesOfLabel, maxTestSamples)) {
                results[count] = Pair(
                    tmpDataArr[labelIndexMapping[label][j]],
                    tmpLabelsArr[labelIndexMapping[label][j]]
                )
                count++
            }
        }
        results.shuffle(random)
        val dataArr = Array(totalExamples) { floatArrayOf() }
        val labelsArr = IntArray(totalExamples)
        results.forEachIndexed { i, pair ->
            dataArr[i] = pair.first
            labelsArr[i] = pair.second
        }
        return Pair(dataArr, labelsArr)
    }

    /*fun createTestBatches(): Array<Array<Array<FloatArray>>> {
        return (0 until HARDataFetcher.NUM_LABELS).map { label ->
            val correspondingDataIndices = sampledLabelsArr.indices
                .filter { i -> sampledLabelsArr[i] == label }
                .take(HARDataFetcher.TEST_BATCH_SIZE)
                .toTypedArray()
            correspondingDataIndices
                .map { i: Int -> featureData[i] }
                .toTypedArray()
        }.toTypedArray()
    }*/

    fun readEntry(i: Int): FloatArray {
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
    }

    fun getEntrySize(): Int {
        return fullDataArr!![0].size
    }

    companion object {
        private var fullDataArr: Array<FloatArray>? = null
        private var fullLabelsArr: IntArray? = null
        private var labelIndexMapping: Array<IntArray>? = null

        @Synchronized
        private fun loadData(baseDirectory: File): Pair<Array<FloatArray>, IntArray> {
            val queue = LinkedList<File>()
            queue.add(baseDirectory)
            val placeholder = floatArrayOf()
            val out = Array(COVIDDataFetcher.NUM_LABELS) { Array(150) { placeholder } }
            val count = IntArray(COVIDDataFetcher.NUM_LABELS)
            while (!queue.isEmpty()) {
                val labelFile = queue.remove()
                val listFiles = labelFile.listFiles()!!
                val label = if (labelFile.name == "healthy") 0 else 1
                for (f in listFiles) {
                    if (f.isDirectory) {
                        queue.add(f)
                    } else if (f.extension == "wav"){
                        out[label][count[label]++] = Wave(f.absolutePath).normalizedAmplitudes
                    }
                }
            }
            val shortestSize = out.flatMap { it.asIterable() }.map { it.size }.minOrNull()!!
            logger.debug { "shortestSize: $shortestSize" }
            val data = Array(150 * COVIDDataFetcher.NUM_LABELS) { floatArrayOf() }
            val labels = IntArray(150 * COVIDDataFetcher.NUM_LABELS)
            var i = 0
            for ((label, arr) in out.withIndex()) {
                for (entry in arr) {
                    data[i] = entry.copyOf(shortestSize)
                    labels[i] = label
                    i++
                }
            }
            return Pair(data, labels)
        }

        @Synchronized
        private fun generateLabelIndexMapping(labelsArr: IntArray): Array<IntArray> {
            return (0 until COVIDDataFetcher.NUM_LABELS).map { label ->
                labelsArr.indices.filter { labelsArr[it] == label }.toIntArray()
            }.toTypedArray()
        }
    }
}
