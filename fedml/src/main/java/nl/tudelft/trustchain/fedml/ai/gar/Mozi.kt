package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import kotlin.math.ceil

private val logger = KotlinLogging.logger("Mozi")

class Mozi(private val fracBenign: Double) : AggregationRule() {
    private val TEST_BATCH = 50

    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomBaseDatasetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): INDArray {
        logger.debug { formatName("MOZI") }
        logger.debug { "Found ${newOtherModels.size} other models" }
        logger.debug { "oldModel: " + oldModel.getDouble(0) }
        val Ndistance = applyDistanceFilter(oldModel, newOtherModels)
        logger.debug { "After distance filter, remaining:${Ndistance.size}" }
        val Nperformance = applyPerformanceFilter(oldModel, Ndistance, network, testDataSetIterator)
        logger.debug { "After performance filter, remaining:${Nperformance.size}" }
//        if (Nperformance.isEmpty()) {
//            logger.debug("Nperformance empty => taking ${Ndistance[0].getDouble(0)}")
//            Nperformance = arrayListOf(Ndistance[0])
//        }

        // This is not included in the original algorithm!!!!
        if (Nperformance.isEmpty()) {
            return oldModel.sub(gradient)
        }

        val Rmozi = average(Nperformance)
        logger.debug("average: ${Rmozi.getDouble(0)}")
        val alpha = 0.5
        val part1 = oldModel.sub(gradient).mul(alpha)
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
        return network.score(sample)
    }

    private fun calculateLoss(
        models: List<INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet
    ): List<Double> {
        val scores = mutableListOf<Double>()
        for (model in models) {
            network.setParameters(model)
            scores.add(network.score(sample))
            logger.debug { "otherLoss = ${scores[scores.size - 1]}" }
        }
        return scores
    }

    private fun applyDistanceFilter(
        oldModel: INDArray,
        newOtherModels: Map<Int, INDArray>
    ): List<INDArray> {
        val distances = hashMapOf<Int, Double>()
        for (otherModel in newOtherModels) {
            logger.debug { "Distance calculated: ${oldModel.distance2(otherModel.value)}" }
            distances[otherModel.key] = oldModel.distance2(otherModel.value)
        }
        val sortedDistances = distances.toList().sortedBy { (_, value) -> value }.toMap()
        val numBenign = ceil(fracBenign * newOtherModels.size).toInt()
        logger.debug { "#benign: $numBenign" }
        return sortedDistances
            .keys
            .take(numBenign)
            .map { newOtherModels[it]!! }
    }

    private fun applyPerformanceFilter(
        oldModel: INDArray,
        newOtherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): List<INDArray> {
        val result = arrayListOf<INDArray>()
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        logger.debug { "oldLoss: $oldLoss" }
        val otherLosses = calculateLoss(newOtherModels, network, sample)
        for ((index, otherLoss) in otherLosses.withIndex()) {
            logger.debug { "otherLoss $index: $otherLoss" }
            if (otherLoss <= oldLoss) {
                result.add(newOtherModels[index])
                logger.debug { "ADDING model($index): " + newOtherModels[index].getDouble(0) }
            } else {
                logger.debug { "NOT adding model($index): " + newOtherModels[index].getDouble(0) }
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

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
