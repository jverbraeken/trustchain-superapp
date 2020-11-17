package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.util.MathUtils
import java.io.File
import java.nio.file.Paths
import java.util.*

fun transposeMatrix(matrix: Array<DoubleArray>): Array<DoubleArray> {
    val m = matrix.size
    val n: Int = matrix[0].size
    val transposedMatrix = Array(n) { DoubleArray(m) }
    for (x in 0 until n) {
        for (y in 0 until m) {
            transposedMatrix[x][y] = matrix[y][x]
        }
    }
    return transposedMatrix
}

class HARDataFetcher(
    baseDirectory: File,
    seed: Long,
    iteratorDistribution: List<Int>,
    dataSetType: DataSetType,
    maxTestSamples: Int,
    behavior: Behaviors
) : CustomBaseDataFetcher() {
    override val testBatches: List<DataSet> by lazy { createTestBatches() }
    val labels: List<String>
        get() = man.getLabels()

    @Transient
    private var man: HARManager
    private var order: IntArray
    private var rng: Random
    private lateinit var featureData: Array<Array<DoubleArray>> // samples => time step => feature

    init {
        val data = arrayListOf<File>()
        val labels: File
        if (dataSetType == DataSetType.TRAIN) {
            val basePath = Paths.get(baseDirectory.path, "train", "Inertial Signals")
            data.add(basePath.resolve("body_acc_x_train.txt").toFile())
            data.add(basePath.resolve("body_acc_y_train.txt").toFile())
            data.add(basePath.resolve("body_acc_z_train.txt").toFile())
            data.add(basePath.resolve("body_gyro_x_train.txt").toFile())
            data.add(basePath.resolve("body_gyro_y_train.txt").toFile())
            data.add(basePath.resolve("body_gyro_z_train.txt").toFile())
            data.add(basePath.resolve("total_acc_x_train.txt").toFile())
            data.add(basePath.resolve("total_acc_y_train.txt").toFile())
            data.add(basePath.resolve("total_acc_z_train.txt").toFile())
            labels = File(File(baseDirectory, "train"), "y_train.txt")
        } else {
            val basePath = Paths.get(baseDirectory.path, "test", "Inertial Signals")
            data.add(basePath.resolve("body_acc_x_test.txt").toFile())
            data.add(basePath.resolve("body_acc_y_test.txt").toFile())
            data.add(basePath.resolve("body_acc_z_test.txt").toFile())
            data.add(basePath.resolve("body_gyro_x_test.txt").toFile())
            data.add(basePath.resolve("body_gyro_y_test.txt").toFile())
            data.add(basePath.resolve("body_gyro_z_test.txt").toFile())
            data.add(basePath.resolve("total_acc_x_test.txt").toFile())
            data.add(basePath.resolve("total_acc_y_test.txt").toFile())
            data.add(basePath.resolve("total_acc_z_test.txt").toFile())
            labels = File(File(baseDirectory, "test"), "y_test.txt")
        }
        man = HARManager(
            data.toTypedArray(),
            labels,
            iteratorDistribution,
            if (dataSetType == DataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
            seed,
            behavior
        )
        totalExamples = man.getNumSamples()
        numOutcomes = NUM_LABELS
        cursor = 0
        inputColumns = NUM_ATTRIBUTES
        order = IntArray(totalExamples)
        for (i in order.indices) {
            order[i] = i
        }
        rng = Random(seed)
        reset()
    }

    override fun reset() {
        cursor = 0
        curr = null
        MathUtils.shuffleArray(order, rng)
    }

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), numOutcomes.toLong())
        if (featureData.size < numExamples) {
            featureData = Array(numExamples) { Array(NUM_TIMESTEPS) { DoubleArray(NUM_DIMENSIONS) } }
        }
        var actualExamples = 0
        var i = 0
        while (i < numExamples) {
            if (!hasMore()) break
            val entries = man.readEntryUnsafe(order[cursor])
            val label = man.readLabel(order[cursor])
            labels.put(actualExamples, label, 1.0f)
            val features = entries.indices.map {
                entries[it] = entries[it].trim()
                val parts = entries[it].split("\\s+").toTypedArray()
                Arrays.stream(parts).mapToDouble { s: String -> s.toDouble() }.toArray()
            }.toTypedArray()
            val timeSteps = transposeMatrix(features)
            featureData[actualExamples] = timeSteps
            actualExamples++
            i++
            cursor++
        }
        val features: INDArray
        features = if (featureData.size == actualExamples) {
            Nd4j.create(featureData)
        } else {
            Nd4j.create(featureData.copyOfRange(0, actualExamples))
        }
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        curr = DataSet(features, labels)
    }

    private fun createTestBatches(): List<DataSet> {
        val testBatches = man.createTestBatches()
        if (featureData.size < testBatches[0].size) {
            featureData = Array(testBatches[0].size) { Array(NUM_TIMESTEPS) { DoubleArray(NUM_DIMENSIONS) } }
        }
        var actualExamples = 0
        val result = arrayListOf<DataSet>()
        for ((label, batch) in testBatches.withIndex()) {
            val labels = Nd4j.zeros(DataType.FLOAT, testBatches[0].size.toLong(), numOutcomes.toLong())
            for (seq in batch) {
                labels.put(actualExamples, label, 1.0f)
                val features = seq.map {
                    val trimmed = it.trim()
                    val parts = trimmed.split("\\s+").toTypedArray()
                    Arrays.stream(parts).mapToDouble { s: String -> s.toDouble() }.toArray()
                }.toTypedArray()
                val timeSteps = transposeMatrix(features)
                featureData[actualExamples] = timeSteps
                actualExamples++
            }
            val features = Nd4j.create(featureData.copyOf())
            result.add(DataSet(features, labels))
        }
        return result
    }

    companion object {
        const val NUM_ATTRIBUTES = 561
        const val NUM_DIMENSIONS = 9
        const val NUM_TIMESTEPS = 128
        const val NUM_LABELS = 6
    }
}
