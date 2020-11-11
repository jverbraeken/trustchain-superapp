package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.stream.Collectors
import kotlin.math.ceil
import kotlin.math.min

private val logger = KotlinLogging.logger("Mozi")

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle(private val fracBenign: Double) : AggregationRule() {
    private val TEST_BATCH = 50

    @ExperimentalStdlibApi
    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>
    ): INDArray {
        logger.debug { formatName("MOZI") }
        logger.debug { "Found ${otherModels.size} other models" }
        logger.debug { "oldModel: " + oldModel.getDouble(0) }
        val newModel = oldModel.sub(gradient)
        val Ndistance: List<INDArray> = applyDistanceFilter(oldModel, newModel, otherModels, allOtherModelsBuffer)
        logger.debug { "After distance filter, remaining:${Ndistance.size}" }
        val Nperformance: List<INDArray> = applyPerformanceFilter(oldModel, Ndistance, network, testDataSetIterator)
        logger.debug { "After performance filter, remaining:${Nperformance.size}" }
//        if (Nperformance.isEmpty()) {
//            logger.debug("Nperformance empty => taking ${Ndistance[0].getDouble(0)}")
//            Nperformance = arrayListOf(Ndistance[0])
//        }

        // This is not included in the original algorithm!!!!
        if (Nperformance.isEmpty()) {
            return oldModel.sub(gradient)
        }

        val Rmozi: INDArray = average(Nperformance)
        logger.debug("average: ${Rmozi.getDouble(0)}")
        val alpha = 0.5
        val part1 = oldModel.mul(alpha)
        val result = part1.add(Rmozi.mul(1 - alpha))
        logger.debug("result: ${result.getDouble(0)}")
        return result
    }

    private fun calculateLoss(
        model: INDArray,
        network: MultiLayerNetwork,
        sample: DataSet
    ): Double {
        network.setParameters(model)
        return network.scoreExamples(sample, true).sumNumber().toDouble()
    }

    private fun calculateLoss(
        models: List<INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet
    ): List<Double> {
        val scores: MutableList<Double> = mutableListOf()
        for (model in models) {
            network.setParameters(model)
            scores.add(network.scoreExamples(sample, true).sumNumber().toDouble())
            logger.debug { "otherLoss = ${scores[scores.size - 1]}" }
        }
        return scores
    }

    private fun applyDistanceFilter(
        oldModel: INDArray,
        newModel: INDArray,
        otherModels: List<INDArray>,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>
    ): List<INDArray> {
        val distances: MutableMap<Int, Double> = hashMapOf()
        for ((index, otherModel) in otherModels.withIndex()) {
            logger.debug { "Distance calculated: ${min(otherModel.distance2(oldModel), otherModel.distance2(newModel))}" }
            distances[index] = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
        }
        for (i in 0 until min(20 - distances.size, allOtherModelsBuffer.size)) {
            val otherModel = allOtherModelsBuffer.elementAt(allOtherModelsBuffer.size - 1 - i)
            distances[1000000 + i] = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
        }
        val sortedDistances: Map<Int, Double> = distances.toList().sortedBy { (_, value) -> value }.toMap()
        val numBenign = ceil(fracBenign * otherModels.size).toLong()
        logger.debug { "#benign: $numBenign" }
        return sortedDistances.keys.stream().limit(numBenign).filter { it < 1000000 }.map { otherModels[it] }
            .collect(Collectors.toList())
    }

    private fun applyPerformanceFilter(
        oldModel: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): List<INDArray> {
        val result: MutableList<INDArray> = arrayListOf()
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        logger.debug { "oldLoss: $oldLoss" }
        val otherLosses = calculateLoss(otherModels, network, sample)
        for ((index, otherLoss) in otherLosses.withIndex()) {
            logger.debug { "otherLoss $index: $otherLoss" }
            if (otherLoss <= oldLoss) {
                result.add(otherModels[index])
                logger.debug { "ADDING model($index): " + otherModels[index].getDouble(0) }
            } else {
                logger.debug { "NOT adding model($index): " + otherModels[index].getDouble(0) }
            }
        }
        return result
    }

    private fun average(list: List<INDArray>): INDArray {
        var arr: INDArray? = null
        for ((index, value) in list.iterator().withIndex()) {
            logger.debug { "Averaging: " + value.getDouble(0) }
            if (index == 0) {
                arr = value
                continue
            }
            arr = arr!!.add(value)
            logger.debug { "Arr = " + arr.getDouble(0) }
        }
        return arr!!.div(list.size)
    }
}
