package nl.tudelft.trustchain.fedml.ai.gar

import mu.KLogger
import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.stream.Collectors
import kotlin.math.ceil
import kotlin.math.min

private val mpl = KotlinLogging.logger("Bristle")
private fun debug(msg: () -> Any?) {
    mpl.debug(msg)
}

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle(private val fracBenign: Double) : AggregationRule() {
    private val TEST_BATCH = 500
    private val NUM_MODELS_TO_EVALUATE = 10
    private val EXPLORATION_RATIO = 0.1

    @ExperimentalStdlibApi
    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>,
        logging: Boolean
    ): INDArray {
        debug { formatName("BRISTLE") }
        debug { "Found ${otherModels.size} other models" }
        debug { "oldModel: ${oldModel.getDouble(0) }" }
        val newModel = oldModel.sub(gradient)
        debug { "newModel: ${newModel.getDouble(0) }" }
        debug { "otherModels: ${otherModels.map { it.getDouble(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(oldModel, newModel, otherModels, allOtherModelsBuffer)
        debug { "distances: $distances"}
        val numCloseModels1 = NUM_MODELS_TO_EVALUATE - NUM_MODELS_TO_EVALUATE * EXPLORATION_RATIO
        val numCloseModels2 = ceil(fracBenign * otherModels.size)
        val numCloseModels = min(numCloseModels1, numCloseModels2).toLong()
        debug { "#numCloseModels: $numCloseModels" }
        val closeModels = distances
            .keys
            .stream()
            .limit(numCloseModels)
            .filter { it < 1000000 }
            .map { otherModels[it] }
            .collect(Collectors.toList())
        debug { "closeModels: ${closeModels.map { it.getDouble(0) }.toCollection(ArrayList())}" }

        val notCloseModels = distances
            .keys
            .stream()
            .skip(numCloseModels)
            .filter { it < 1000000 }
            .map { otherModels[it] }
            .collect(Collectors.toList())
        notCloseModels.shuffle()
        notCloseModels.take(NUM_MODELS_TO_EVALUATE - closeModels.size)  // perhaps not enough elements...
        debug { "notCloseModels: ${notCloseModels.map { it.getDouble(0) }.toCollection(ArrayList())}" }

        val combinedModels = listOf(closeModels, notCloseModels).flatten()
        debug { "combinedModels: ${combinedModels.map { it.getDouble(0) }.toCollection(ArrayList())}" }
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        debug { "oldLoss: $oldLoss" }
        val losses = calculateLosses(combinedModels, network, sample)
        debug { "losses: $losses"}
        val oldLoss2 = calculateLoss(oldModel, network, sample)
        debug { "oldLoss2: $oldLoss2" }
        val modelsToWeight = mapLossesToWeight(losses, oldLoss)
        debug { "modelsToWeight: $modelsToWeight"}

        return if (modelsToWeight.isEmpty()) {
            newModel
        } else {
            weightedAverage(modelsToWeight, combinedModels, newModel)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        oldModel: INDArray,
        newModel: INDArray,
        otherModels: List<INDArray>,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels.withIndex()) {
            debug { "Distance calculated: ${min(otherModel.distance2(oldModel), otherModel.distance2(newModel))}" }
            distances[index] = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
        }
        for (i in 0 until min(20 - distances.size, allOtherModelsBuffer.size)) {
            val otherModel = allOtherModelsBuffer.elementAt(allOtherModelsBuffer.size - 1 - i)
            distances[1000000 + i] = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
        }
        return distances.toList().sortedBy { (_, value) -> value }.toMap()
    }

    private fun calculateLoss(
        model: INDArray,
        network: MultiLayerNetwork,
        sample: DataSet
    ): Double {
        network.setParameters(model)
        return network.score(sample)
    }

    private fun calculateLosses(
        models: List<INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet
    ): Map<Int, Double> {
        val scores = mutableMapOf<Int, Double>()
        for ((index, model) in models.withIndex()) {
            network.setParameters(model)
            scores[index] = network.score(sample)
        }
        return scores
    }

    private fun mapLossesToWeight(otherLosses: Map<Int, Double>, oldLoss: Double): Map<Int, Double> {
        val minWeight = 1
        val maxWeight = 5
        return otherLosses
            .filter { it.value < oldLoss }
            .map { (index, loss) ->
                val weightDistance = maxWeight - minWeight
                val performance = (1 - (loss / oldLoss))
                Pair(index, minWeight + weightDistance * performance)
            }
            .toMap()
    }

    private fun weightedAverage(modelsToWeight: Map<Int, Double>, otherModels: List<INDArray>, newModel: INDArray): INDArray {
        var arr: INDArray? = null
        for ((index, weight) in modelsToWeight.entries) {
            if (index == 0) {
                arr = otherModels[index].mul(weight)
                continue
            }
            arr = arr!!.add(otherModels[index].mul(weight))
        }
        arr = arr!!.add(newModel)
        val totalWeight = modelsToWeight.values.sum() + 1  // + 1 for the new model
        debug { "totalWeight: $totalWeight"}
        debug { "weightedAverage: ${arr!!.div(totalWeight)}" }
        return arr!!.div(totalWeight)
    }
}
