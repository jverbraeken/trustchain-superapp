package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("Median")

class Bridge(val b: Int) : AggregationRule() {
    override fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        gradient: INDArray,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int> {
        val models: MutableList<INDArray> = arrayListOf(myModel.first)
        otherModelPairs.forEach { models.add(it.first) }
        logger.debug { "Found ${models.size} models in total" }
        return if (models.size == 1) {
            Pair(myModel.first.sub(gradient), 999999)
        } else {
            val modelsAsArrays = models.map { it.toDoubleMatrix()[0] }
            val newMatrix = Array(1) { DoubleArray(modelsAsArrays[0].size) }
            for (i in modelsAsArrays[0].indices) {
                val elements = ArrayList<Double>(modelsAsArrays.size)
                modelsAsArrays.forEach { elements.add(it[i]) }
                newMatrix[0][i] = trimmedMean(b, elements)
            }
            Pair(NDArray(newMatrix).sub(gradient), 999999)
        }
    }
}
