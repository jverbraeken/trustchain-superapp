package nl.tudelft.trustchain.fedml.ai.dataset.covid

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDataFetcher
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.io.File

private val logger = KotlinLogging.logger("HARDataFetcher")

class COVIDDataFetcher(
    baseDirectory: File,
    seed: Long,
    iteratorDistribution: IntArray,
    dataSetType: CustomDataSetType,
    maxTestSamples: Int,
    behavior: Behaviors,
) : CustomBaseDataFetcher(seed) {
    override val testBatches by lazy { arrayOf<DataSet?>() }
    val labels: List<String>
        get() = man.getLabels()

    @Transient
    private var man: COVIDManager

    /**  samples => time step => feature **/
    private var featureData = Array(1) { arrayOf<FloatArray>() }

    init {
        man = COVIDManager(
            baseDirectory,
            iteratorDistribution,
            if (dataSetType == CustomDataSetType.TRAIN || dataSetType == CustomDataSetType.FULL_TRAIN) Int.MAX_VALUE else maxTestSamples,
            seed,
            behavior
        )
        totalExamples = man.getNumSamples()
        shortestSize = man.getEntrySize()
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
            featureData = Array(numExamples) { arrayOf() }
        }
        var actualExamples = 0
        var i = 0
        while (i < numExamples) {
            if (!hasMore()) break
            val label = man.readLabel(order[cursor])
            labels.put(actualExamples, label, 1.0f)
            featureData[actualExamples] = Array(1) { man.readEntry(order[cursor]) }
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

    /*private fun createTestBatches(): Array<DataSet?> {
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
    }*/

    /*private fun createTestBatch(label: Int, batch: Array<Array<FloatArray>>): DataSet {
        val labels = Nd4j.zeros(DataType.FLOAT, batch.size.toLong(), numOutcomes.toLong())
        for ((i, seq) in batch.withIndex()) {
            labels.put(i, label, 1.0f)
            featureData[i] = seq
        }
        val features = Nd4j.create(featureData)
        return DataSet(features, labels)
    }*/

    companion object {
        var shortestSize = -1
        const val NUM_DIMENSIONS = 1
        const val NUM_LABELS = 2
        const val TEST_BATCH_SIZE = 20
    }
}
