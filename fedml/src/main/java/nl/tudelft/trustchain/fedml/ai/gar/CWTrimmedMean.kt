package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

private val logger = KotlinLogging.logger("Median")

fun trimmedMean(b: Int, l: List<Double>) = l.sorted().subList(b, l.size - b - 1).average()

class CWTrimmedMean(val b: Int) : AggregationRule() {
    override fun integrateParameters(
        myModel: Pair<INDArray, Int>,
        gradient: INDArray,
        otherModelPairs: List<Pair<INDArray, Int>>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator
    ): Pair<INDArray, Int> {
        val models: MutableList<INDArray> = arrayListOf(myModel.first.sub(gradient))
        otherModelPairs.forEach { models.add(it.first) }
        logger.debug { "Found ${models.size} models in total" }
        return if (models.size == 1) {
            Pair(models[0], 999999)
        } else {
            val modelsAsArrays = models.map { it.toDoubleMatrix()[0] }
            val newMatrix = Array(1) { DoubleArray(modelsAsArrays[0].size) }
            for (i in modelsAsArrays[0].indices) {
                val elements = ArrayList<Double>(modelsAsArrays.size)
                modelsAsArrays.forEach { elements.add(it[i]) }
                newMatrix[0][i] = trimmedMean(b, elements)
            }
            Pair(NDArray(newMatrix), 999999)
        }
    }
}
