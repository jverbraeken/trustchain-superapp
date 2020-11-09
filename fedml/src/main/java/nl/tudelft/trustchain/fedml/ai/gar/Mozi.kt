package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.stream.Collectors
import kotlin.math.ceil

private val logger = KotlinLogging.logger("Mozi")

class Mozi(private val fracBenign: Double) : AggregationRule() {
    private val TEST_BATCH = 50

    override fun integrateParameters(
        myModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): INDArray {
        logger.debug { formatName("MOZI") }
        val otherModels: MutableList<INDArray> = arrayListOf()
        otherModels.forEach { otherModels.add(it) }
        logger.debug { "Found ${otherModels.size} other models" }
        logger.debug { "MyModel: " + myModel.getDouble(0) }
        val Ndistance: List<INDArray> = applyDistanceFilter(myModel, otherModels)
        logger.debug { "After distance filter, remaining:${Ndistance.size}" }
        val Nperformance: List<INDArray> = applyPerformanceFilter(myModel, Ndistance, network, testDataSetIterator)
        logger.debug { "After performance filter, remaining:${Nperformance.size}" }
//        if (Nperformance.isEmpty()) {
//            logger.debug("Nperformance empty => taking ${Ndistance[0].getDouble(0)}")
//            Nperformance = arrayListOf(Ndistance[0])
//        }

        // This is not included in the original algorithm!!!!
        if (Nperformance.isEmpty()) {
            return myModel.sub(gradient)
        }

        val Rmozi: INDArray = average(Nperformance)
        logger.debug("average: ${Rmozi.getDouble(0)}")
        val alpha = 0.5
        val part1 = myModel.mul(alpha)
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
        myModel: INDArray,
        otherModels: List<INDArray>
    ): List<INDArray> {
        val distances: MutableMap<Double, INDArray> = hashMapOf()
        for (otherModel in otherModels) {
            logger.debug { "Distance calculated: ${myModel.distance2(otherModel)}" }
            distances[myModel.distance2(otherModel)] = otherModel
        }
        val sortedDistances: Map<Double, INDArray> = distances.toSortedMap()
        val numBenign = ceil(fracBenign * otherModels.size).toLong()
        logger.debug { "#benign: $numBenign" }
        return sortedDistances.values.stream().limit(numBenign).collect(Collectors.toList())
    }

    private fun applyPerformanceFilter(
        myModel: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): List<INDArray> {
        val result: MutableList<INDArray> = arrayListOf()
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val myLoss = calculateLoss(myModel, network, sample)
        logger.debug { "myLoss: $myLoss" }
        val otherLosses = calculateLoss(otherModels, network, sample)
        for ((index, otherLoss) in otherLosses.withIndex()) {
            logger.debug { "otherLoss $index: $otherLoss" }
            if (otherLoss <= myLoss) {
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
