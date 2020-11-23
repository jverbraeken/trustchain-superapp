package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.stream.Collectors
import kotlin.math.max
import kotlin.math.min

private val mpl = KotlinLogging.logger("Bristle")
private fun debug(logging: Boolean, msg: () -> Any?) {
    if (logging) mpl.debug(msg)
}

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle : AggregationRule() {
    private val TEST_BATCH = 500
    private val NUM_MODELS_EXPLOITATION = 9
    private val NUM_MODELS_EXPLORATION = 1

    @ExperimentalStdlibApi
    override fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
        testBatches: List<DataSet?>,
        countPerPeer: Map<Int, Int>
    ): INDArray {
        debug(logging) { formatName("BRISTLE") }
        debug(logging) { "Found ${otherModels.size} other models" }
        debug(logging) { "oldModel: ${oldModel.getDouble(0)}" }
        val newModel = oldModel.sub(gradient)
        debug(logging) { "newModel: ${newModel.getDouble(0)}" }
        debug(logging) { "otherModels: ${otherModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(oldModel, newModel, otherModels, allOtherModelsBuffer, logging)
        debug(logging) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .stream()
            .limit(NUM_MODELS_EXPLOITATION.toLong())
            .filter { it < 1000000 }
            .map { Pair(it, otherModels[it]!!) }
            .collect(Collectors.toList())
            .toMap()
        debug(logging) { "closeModels: ${exploitationModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val explorationModelsList = distances
            .keys
            .stream()
            .skip(NUM_MODELS_EXPLOITATION.toLong())
            .filter { it < 1000000 }
            .map { Pair(it, otherModels[it]!!) }
            .collect(Collectors.toList())
        explorationModelsList.shuffle()
        explorationModelsList.take(NUM_MODELS_EXPLORATION)
        val explorationModels = explorationModelsList.toMap()
        debug(logging) { "notCloseModels: ${explorationModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Int, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
        debug(logging) { "combinedModels: ${combinedModels.map { it.value.getDouble(0) }.toCollection(ArrayList())}" }
        testDataSetIterator.reset()
        val sample = testDataSetIterator.next(TEST_BATCH)
        val oldLoss = calculateLoss(oldModel, network, sample)
        debug(logging) { "oldLoss: $oldLoss" }
        val oldLossPerClass = calculateLossPerClass(oldModel, network, testBatches)
        debug(logging) { "oldLossPerClass: ${oldLossPerClass.toList()}" }
        val losses = calculateLosses(combinedModels, network, sample)
        debug(logging) { "losses: $losses" }
        val lossesPerClass = calculateLossesPerClass(combinedModels, network, testBatches)
        debug(logging) { "lossesPerClass: ${lossesPerClass.map { Pair(it.key, it.value.toList()) }}" }
        val modelsToWeight = mapLossesToWeight(losses, oldLoss)
        debug(logging) { "modelsToWeight: $modelsToWeight" }
        val modelsPerClassToWeight = mapLossesPerClassToWeight(lossesPerClass, oldLossPerClass, countPerPeer, logging)
        debug(logging) { "modelsPerClassToWeight: $modelsPerClassToWeight" }

        return if (modelsPerClassToWeight.isEmpty()) {
            newModel
        } else {
            weightedAverage(modelsPerClassToWeight, combinedModels, newModel, logging)
        }
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        oldModel: INDArray,
        newModel: INDArray,
        otherModels: Map<Int, INDArray>,
        allOtherModelsBuffer: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            debug(logging) { "Distance calculated: ${min(otherModel.distance2(oldModel), otherModel.distance2(newModel))}" }
            distances[index] = min(otherModel.distance2(oldModel), otherModel.distance2(newModel))
        }
        for (i in 0 until min(20 - distances.size, allOtherModelsBuffer.size)) {
            val otherModel = allOtherModelsBuffer.elementAt(allOtherModelsBuffer.size - 1 - i)
            distances[1000000 + otherModel.first] =
                min(otherModel.second.distance2(oldModel), otherModel.second.distance2(newModel))
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

    private fun calculateLossPerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testBatches: List<DataSet?>
    ): Array<Double?> {
        network.setParameters(model)
        return testBatches
            .map { if (it == null) null else network.score(it) }
            .toTypedArray()
    }

    private fun calculateLosses(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        sample: DataSet
    ): Map<Int, Double> {
        return models.map { (index, model) ->
            network.setParameters(model)
            Pair(index, network.score(sample))
        }.toMap()
    }

    private fun calculateLossesPerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testBatches: List<DataSet?>
    ): Map<Int, Array<Double?>> {
        return models.map { (index, model) ->
            Pair(
                index, testBatches.map { testBatch ->
                    if (testBatch == null) {
                        null
                    } else {
                        network.setParameters(model)
                        network.score(testBatch)
                    }
                }.toTypedArray()
            )
        }.toMap()
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

    private fun mapLossesPerClassToWeight(
        otherLossesPerClass: Map<Int, Array<Double?>>,
        oldLossPerClass: Array<Double?>,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): Map<Int, Double> {
        val smallestLossPerPeer = otherLossesPerClass
            .map { (peer, lossPerClass) ->
                Pair(peer, lossPerClass
                    .mapIndexed { label, loss -> Pair(label, loss ?: Double.MAX_VALUE) }
                    .sortedBy { it.second }
                    .take(countPerPeer[peer]!!)
                    .toMap())
            }.toMap()
        debug(logging) { "smallestLossPerPeer: $smallestLossPerPeer" }

        return smallestLossPerPeer.map { (peer, lossPerLabel) ->
            val smallestOtherLoss = lossPerLabel.map { (_, smallestLosses) -> smallestLosses }.sum()
            val smallestOwnLoss = lossPerLabel.map { (label, _) -> oldLossPerClass[label]!!}.sum()
            Pair(peer, max(0.0, 4.0 + (4 - 4 * (smallestOtherLoss / smallestOwnLoss))))
        }.toMap()
    }

    private fun weightedAverage(
        modelsToWeight: Map<Int, Double>,
        otherModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean
    ): INDArray {
        var arr: INDArray? = null
        modelsToWeight.onEachIndexed { indexAsNum, (indexAsPeer, weight) ->
            if (indexAsNum == 0) {
                arr = otherModels[indexAsPeer]!!.mul(weight)
            } else {
                arr = arr!!.add(otherModels[indexAsPeer]!!.mul(weight))
            }
        }
        arr = arr!!.add(newModel)
        val totalWeight = modelsToWeight.values.sum() + 1  // + 1 for the new model
        debug(logging) { "totalWeight: $totalWeight" }
        debug(logging) { "weightedAverage: ${arr!!.div(totalWeight)}" }
        return arr!!.div(totalWeight)
    }
}
