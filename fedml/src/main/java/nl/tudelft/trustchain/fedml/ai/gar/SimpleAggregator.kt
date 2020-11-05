package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("SimpleAggregator")

class SimpleAggregator : AggregationRule() {
    override fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        gradient: INDArray,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int> {
        val allModels = otherModelPairs.toMutableList()
        allModels.add(myModel)
        val totalWeight = allModels.map { it.second }.reduce { sum, elem -> sum + elem }
        var arr: INDArray =
            allModels[0].first.mul(allModels[0].second.toDouble() / totalWeight.toDouble())
        for (i in 1 until allModels.size) {
            arr = arr.add(allModels[i].first.mul(allModels[i].second.toDouble() / totalWeight.toDouble()))
        }
        return Pair(arr.sub(gradient), totalWeight)
    }
}
