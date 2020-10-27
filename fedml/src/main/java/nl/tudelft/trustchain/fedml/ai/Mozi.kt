package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.stream.Collectors
import kotlin.math.ceil

class Mozi : AggregationRule() {
    private val FRAC_BENIGN = 0.5
    private val TEST_BATCH = 50

    override fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int> {
        val otherModels : MutableList<INDArray> = arrayListOf()
        for (otherModelPair in otherModelPairs) {
            otherModels.add(otherModelPair.first)
        }
        val Ndistance: List<INDArray> = applyDistanceFilter(myModel, otherModels)
        val Nperformance: List<INDArray> = applyPerformanceFilter(myModel, Ndistance, network, testDataSetIterator)
        val Rmozi: INDArray = average(Nperformance)
        val alpha = 0.5
        val part1 = myModel.first.mul(alpha)
        return Pair(part1.add(Rmozi.mul(1 - alpha)), 999999)
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
        val scores : MutableList<Double> = mutableListOf()
        for (model in models) {
            network.setParameters(model)
            scores.add(network.scoreExamples(sample, true).sumNumber().toDouble())
        }
        return scores
    }

    private fun applyDistanceFilter(
        myModel: Pair<INDArray, Int>,
        otherModels: List<INDArray>
    ): List<INDArray> {
        val distances: MutableMap<Double, INDArray> = hashMapOf()
        for (otherModel in otherModels) {
            distances[myModel.first.distance2(otherModel)] = otherModel
        }
        val sortedDistances: Map<Double, INDArray> = distances.toSortedMap()
        val numBenign = ceil(FRAC_BENIGN * otherModels.size).toLong()
        return sortedDistances.values.stream().limit(numBenign).collect(Collectors.toList())
    }

    private fun applyPerformanceFilter(myModel: Pair<INDArray, Int>, otherModels: List<INDArray>, network: MultiLayerNetwork, testDataSetIterator: DataSetIterator): List<INDArray> {
        val result : MutableList<INDArray> = arrayListOf()
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val myLoss = calculateLoss(myModel.first, network, sample)
        val otherLosses = calculateLoss(otherModels, network, sample)
        for ((index, otherLoss) in otherLosses.withIndex()) {
            if (otherLoss <= myLoss) {
                result.add(otherModels[index])
            }
        }
        if (result.isEmpty()) {
            result.add(otherModels[otherLosses.indexOf(otherLosses.minBy { it })])
        }
        return result
    }

    private fun average(list: List<INDArray>): INDArray {
        var arr: INDArray? = null
        for ((index, value) in list.iterator().withIndex()) {
            if (index == 0) {
                arr = value
                continue
            }
            arr!!.add(value)
        }
        return arr!!.div(list.size)
    }
}
