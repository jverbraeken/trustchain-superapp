package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

private val logger = KotlinLogging.logger("Krum")

fun getKrum(models: Array<INDArray>, b: Int): Int {
    val distances = Array(models.size) { DoubleArray(models.size) }
    for (i in 0 until models.size) {
        distances[i][i] = 9999999.0
        for (j in i + 1 until models.size) {
            val distance = models[i].distance2(models[j])
            distances[i][j] = distance
            distances[j][i] = distance
        }
    }
    val summedDistances = distances.map {
        val sorted = it.sorted()
        sorted.take(models.size - b - 2 - 1).sum()  // The additional -1 is because a peer is not a neighbor of itself
    }.toTypedArray()
    return summedDistances.indexOf(summedDistances.minOrNull()!!)
}

class Krum(private val b: Int) : AggregationRule() {
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): INDArray {
        debug(logging) { formatName("Krum") }
        val modelMap = HashMap<Int, INDArray>()
        val newModel = oldModel.sub(gradient)
        modelMap[-1] = newModel
        modelMap.putAll(newOtherModels)
        val models = modelMap.values.toTypedArray()
        debug(logging) { "Found ${models.size} models in total" }
        return if (models.size <= b + 2 + 1) {  // The additional +1 is because we need to add the current peer itself
            debug(logging) { "Not using KRUM rule because not enough models found..." }
            newModel
        } else {
            val bestCandidate = getKrum(models, b)
            newModel.addi(models[bestCandidate]).divi(2)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
