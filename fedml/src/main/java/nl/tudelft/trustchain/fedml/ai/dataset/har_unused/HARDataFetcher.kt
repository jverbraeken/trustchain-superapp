package nl.tudelft.trustchain.fedml.ai.dataset.har_unused

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File
import java.nio.file.Paths

private val logger = KotlinLogging.logger("HARDataFetcher")

fun transposeMatrix(matrix: Array<Array<Float>>): Array<FloatArray> {
    val m = matrix.size
    val n = matrix[0].size
    val transposedMatrix = Array(n) { FloatArray(m) }
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
    iteratorDistribution: IntArray,
    dataSetType: CustomDataSetType,
    maxTestSamples: Int,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { createTestBatches() }
    val labels: List<String>
        get() = man.getLabels()

    @Transient
    private var man: HARManager

    /**  samples => time step => feature **/
    private var featureData = Array(1) { Array(NUM_TIMESTEPS) { FloatArray(NUM_DIMENSIONS) } }

    init {
        val data = arrayListOf<File>()
        val labels: File
        if (dataSetType == CustomDataSetType.TRAIN) {
            val basePath = Paths.get(baseDirectory.path, "train", "Inertial Signals")
            data.add(basePath.resolve("body_acc_x_train.txt").toFile())
            data.add(basePath.resolve("body_acc_y_train.txt").toFile())
            data.add(basePath.resolve("body_acc_z_train.txt").toFile())/*
            data.add(basePath.resolve("body_gyro_x_train.txt").toFile())
            data.add(basePath.resolve("body_gyro_y_train.txt").toFile())
            data.add(basePath.resolve("body_gyro_z_train.txt").toFile())
            data.add(basePath.resolve("total_acc_x_train.txt").toFile())
            data.add(basePath.resolve("total_acc_y_train.txt").toFile())
            data.add(basePath.resolve("total_acc_z_train.txt").toFile())*/
            labels = File(File(baseDirectory, "train"), "y_train.txt")
        } else {
            val basePath = Paths.get(baseDirectory.path, "test", "Inertial Signals")
            data.add(basePath.resolve("body_acc_x_test.txt").toFile())
            data.add(basePath.resolve("body_acc_y_test.txt").toFile())
            data.add(basePath.resolve("body_acc_z_test.txt").toFile())/*
            data.add(basePath.resolve("body_gyro_x_test.txt").toFile())
            data.add(basePath.resolve("body_gyro_y_test.txt").toFile())
            data.add(basePath.resolve("body_gyro_z_test.txt").toFile())
            data.add(basePath.resolve("total_acc_x_test.txt").toFile())
            data.add(basePath.resolve("total_acc_y_test.txt").toFile())
            data.add(basePath.resolve("total_acc_z_test.txt").toFile())*/
            labels = File(File(baseDirectory, "test"), "y_test.txt")
        }
        man = HARManager(
            data.toTypedArray(),
            labels,
            iteratorDistribution,
            if (dataSetType == CustomDataSetType.TRAIN) Int.MAX_VALUE else maxTestSamples,
            seed,
            behavior
        )
        totalExamples = man.getNumSamples()
        numOutcomes = NUM_LABELS
        cursor = 0
        order = IntArray(totalExamples)
        for (i in order.indices) {
            order[i] = i
        }
        reset()
    }

    override fun fetch(numExamples: Int) {
        check(hasMore()) { "Unable to get more" }
        var labels = Nd4j.zeros(DataType.FLOAT, numExamples.toLong(), numOutcomes.toLong())
        if (featureData.size != numExamples) {
            featureData = Array(numExamples) { Array(NUM_TIMESTEPS) { FloatArray(NUM_DIMENSIONS) } }
        }
        var actualExamples = 0
        var i = 0
        while (i < numExamples) {
            if (!hasMore()) break
            val label = man.readLabel(order[cursor])
            labels.put(actualExamples, label, 1.0f)
            featureData[actualExamples] = transpose(man.readEntry(order[cursor]))
            actualExamples++
            i++
            cursor++
        }
        val features = Nd4j.create(
            if (featureData.size == actualExamples) featureData
            else featureData.copyOfRange(0, actualExamples)
        )
        if (actualExamples < numExamples) {
            labels = labels[NDArrayIndex.interval(0, actualExamples), NDArrayIndex.all()]
        }
        curr = DataSet(features, labels)
    }

    private fun createTestBatches(): Array<DataSet?> {
        val testBatches = man.createTestBatches()
        if (featureData.size != TEST_BATCH_SIZE) {
            featureData = Array(TEST_BATCH_SIZE) { Array(NUM_TIMESTEPS) { FloatArray(NUM_DIMENSIONS) } }
        }
        val result = arrayListOf<DataSet?>()
        for ((label, batch) in testBatches.withIndex()) {
            result.add(
                if (batch.isEmpty()) null
                else createTestBatch(label, batch)
            )
        }
        return result.toTypedArray()
    }

    private fun createTestBatch(label: Int, batch: Array<Array<FloatArray>>): DataSet {
        val labels = Nd4j.zeros(DataType.FLOAT, batch.size.toLong(), numOutcomes.toLong())
        for ((i, seq) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            featureData[i] = seq
        }
        val features = Nd4j.create(featureData)
        return DataSet(features, labels)
    }

    private fun transpose(second: Array<FloatArray>): Array<FloatArray> {
        val result = Array(second[0].size) { FloatArray(second.size) }
        for (i in second.indices) {
            for (j in second[i].indices) {
                result[j][i] = second[i][j]
            }
        }
        return result
    }

    companion object {
        const val NUM_DIMENSIONS = 3
        const val NUM_TIMESTEPS = 50
        const val NUM_LABELS = 6
        const val TEST_BATCH_SIZE = 20
    }
}
