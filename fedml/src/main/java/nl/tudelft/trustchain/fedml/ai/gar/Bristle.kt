package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.NDArrayStrings2
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.math.*

private const val NUM_MODELS_EXPLOITATION = 20
private const val NUM_MODELS_EXPLORATION = 5
private const val MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING = 20

private typealias Peer = Int
private typealias Class = Int
private typealias Score = Double
private typealias Weight = Double
private typealias Certainty = Double

private val logger = KotlinLogging.logger("Bristle")

/**
 * (practical yet robust) byzantine-resilient decentralized stochastic federated learning
 *
 *
 * byzantine-resilient decentralized stochastic gradient descent federated learning, non i.i.d., history-sensitive (= more robust), practical
 */
class Bristle : AggregationRule() {
    @ExperimentalStdlibApi
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): INDArray {
        val formatter = NDArrayStrings2()
        logger.d(true) { formatName("BRISTLE") }
        logger.d(true) { "Found ${newOtherModels.size} other models" }
        logger.d(true) { "oldModel: ${formatter.format(oldModel)}" }
        val newModel = oldModel.sub(gradient)
        logger.d(true) { "newModel: ${formatter.format(newModel)}" }
        logger.d(true) { "otherModels: ${newOtherModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val startTime = System.currentTimeMillis()
        val distances = getDistances(newModel, newOtherModels, recentOtherModels, true)
        logger.d(true) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .take(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .toMap()
        logger.d(true) { "closeModels: ${exploitationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val explorationModels = distances
            .keys
            .drop(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .shuffled()
            .take(NUM_MODELS_EXPLORATION)
            .toMap()
        logger.d(true) { "notCloseModels: ${explorationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Peer, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
//        combinedModels.forEach {
//            logger.d(true) { "combinedModel (${it.key}): ${formatter.format(it.value)}" }
//        }
        val peers = combinedModels.keys.sorted().toIntArray()
        val endTime = System.currentTimeMillis()
        logger.d(true) { "Timing 0: ${endTime - startTime}" }
        testDataSetIterator.reset()
        val startTime2 = System.currentTimeMillis()

        val familiarClasses = testDataSetIterator.labels.map { it.toInt() }
        logger.d(true) { "familiarClasses: $familiarClasses" }
        val foreignLabels = (0 until newModel.columns()).subtract(familiarClasses).toList()
        logger.d(true) { "foreignLabels: $foreignLabels" }

        val myF1s = calculateF1PerClass(newModel, network, testDataSetIterator, familiarClasses)
        logger.d(true) { "myF1s: ${myF1s.toList()}" }

        val f1sPerPeer = calculateF1PerClass(combinedModels, network, testDataSetIterator, familiarClasses)
        logger.d(true) { "f1sPerPeer: ${f1sPerPeer.map { Pair(it.key, it.value.toList()) }}}" }

        val selectedClassesPerPeer = getBestPerformingClassesPerPeer(peers, myF1s, f1sPerPeer, countPerPeer)
        logger.d(true) { "selectedClassesPerPeer: ${selectedClassesPerPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedDiffsPerSelectedPeer = getWeightedDiffsPerSelectedPeer(peers, myF1s, f1sPerPeer, familiarClasses)
        logger.d(true) { "weightedDiffPerSelectedPeer: ${weightedDiffsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val scoresPerPeer = getScoresPerPeer(peers, myF1s, f1sPerPeer, familiarClasses, weightedDiffsPerSelectedPeer, true)
        logger.d(true) { "scoresPerSelectedPeer: ${scoresPerPeer.map { Pair(it.key, it.value.toList()) }}" }



        val certaintyPerSelectedPeer = getCertaintyPerSelectedPeer(peers, f1sPerPeer, selectedClassesPerPeer)
        logger.d(true) { "certaintyPerSelectedPeer: $certaintyPerSelectedPeer" }



        val weightsPerSelectedPeer = getWeightsPerSelectedPeer(peers, scoresPerPeer, certaintyPerSelectedPeer)
        logger.d(true) { "weightsPerSelectedPeer: ${weightsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedAverage = weightedAverage(weightsPerSelectedPeer, combinedModels, newModel, true, familiarClasses)
        logger.d(true) { "weightedAverage: ${formatter.format(weightedAverage)}" }


        val unboundedScoresPerSelectedPeer = getScoresPerPeer(peers, myF1s, f1sPerPeer, familiarClasses, weightedDiffsPerSelectedPeer, false)
        logger.d(true) { "unboundedScoresPerSelectedPeer: ${unboundedScoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightPerSelectedPeer = getWeightPerSelectedPeer(peers, unboundedScoresPerSelectedPeer, certaintyPerSelectedPeer)
        logger.d(true) { "weightPerSelectedPeer: $weightPerSelectedPeer" }

        val result = incorporateForeignWeights(peers, weightPerSelectedPeer, combinedModels, weightedAverage, foreignLabels, true)
        logger.d(true) { "result: ${formatter.format(result)}" }
        val endTime2 = System.currentTimeMillis()
        logger.d(true) { "Timing 1: ${endTime2 - startTime2}" }
        return result
    }

    override fun isDirectIntegration(): Boolean {
        return true
    }

    private fun getDistances(
        newModel: INDArray,
        otherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        logging: Boolean,
    ): Map<Int, Double> {
        val distances = hashMapOf<Int, Double>()
        for ((index, otherModel) in otherModels) {
            val min = otherModel.distance2(newModel)
            logger.d(true) { "Distance calculated: $min" }
            distances[index] = min
        }
        for (i in 0 until min(MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING - distances.size, recentOtherModels.size)) {
            val otherModel = recentOtherModels.elementAt(recentOtherModels.size - 1 - i)
            distances[1100000 + otherModel.first] = otherModel.second.distance2(newModel)
        }
        return distances.toList().sortedBy { (_, value) -> value }.toMap()
    }

    private fun calculateF1PerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
        familiarClasses: List<Class>,
    ): Map<Class, Double> {
        val tw = network.outputLayer.paramTable()
        for (index in 0 until tw.getValue("W").columns()) {
            tw.getValue("W").putColumn(index, model.getColumn(index.toLong()).dup())
        }
        network.outputLayer.setParamTable(tw)
        val evaluations = arrayOf(Evaluation())
        network.doEvaluation(testDataSetIterator, *evaluations)
        return familiarClasses.map { clz ->
            Pair(clz, evaluations[0].f1(clz))
        }.toMap()
    }

    private fun calculateF1PerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
        familiarClasses: List<Class>,
    ): Map<Peer, Map<Class, Double>> {
        return models.map { (index, model) ->
            Pair(index, calculateF1PerClass(model, network, testDataSetIterator, familiarClasses))
        }.toMap()
    }

    private fun getBestPerformingClassesPerPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        countPerPeer: Map<Int, Int>
    ): Map<Peer, IntArray> {
        return peers.map { peer ->
            val selectionSize = countPerPeer.getOrDefault(peer, myF1PerClass.size)
            val selectedClasses = if (selectionSize == peerF1PerClass.getValue(peer).size)
                (0 until selectionSize).toList().toIntArray()
            else
                peerF1PerClass.getValue(peer)
                    .toList()
                    .sortedByDescending { it.second }
                    .take(selectionSize)  // Attackers have a negative index => assume that they have the same classes as the current peer to maximize attack strength
                    .map { it.first }
                    .toIntArray()
            Pair(peer, selectedClasses)
        }.toMap()
    }

    private fun getWeightedDiffsPerSelectedPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        familiarClasses: List<Class>,
    ): Map<Peer, Map<Class, Double>> {
        return peers.map { peer ->
            val weightDiffsPerFamiliarClass = familiarClasses.map { familiarClass ->
                Pair(familiarClass, abs(myF1PerClass.getValue(familiarClass) - peerF1PerClass.getValue(peer).getValue(familiarClass)) * 10)
            }.toMap()
            Pair(peer, weightDiffsPerFamiliarClass)
        }.toMap()
    }

    private fun getScoresPerPeer(
        peers: IntArray,
        myF1PerClass: Map<Class, Double>,
        peerF1PerClass: Map<Peer, Map<Class, Double>>,
        familiarClasses: List<Class>,
        weightedDiffPerSelectedPeer: Map<Peer, Map<Class, Double>>,
        bounded: Boolean
    ): Map<Peer, Map<Class, Score>> {
        return peers.map { peer ->
            val scorePerFamiliarClass = familiarClasses.map { familiarClass ->
                val myF1 = myF1PerClass.getValue(familiarClass)
                val peerF1 = peerF1PerClass.getValue(peer).getValue(familiarClass)
                val weightedDiff = weightedDiffPerSelectedPeer.getValue(peer).getValue(familiarClass)
                val score = if (peerF1 > myF1) weightedDiff.pow(3 + myF1) else -1.0 * weightedDiff.pow(4 + myF1)
                Pair(familiarClass, if (bounded) max(-100.0, score) else score)
            }.toMap()
            Pair(peer, scorePerFamiliarClass)
        }.toMap()
    }

    private fun getWeightsPerSelectedPeer(
        peers: IntArray,
        scoresPerPeer: Map<Peer, Map<Class, Score>>,
        certaintyPerSelectedPeer: Map<Peer, Double>
    ): Map<Peer, Map<Class, Weight>> {
        return peers.map { peer ->
            val weightPerFamiliarClass = scoresPerPeer.getValue(peer).map { (familiarClass, score) ->
                var sigmoid = 1 / (1 + exp(-score / 100))  // sigmoid function
                sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
                sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
                sigmoid = max(0.0, sigmoid)  // no negative weights
                sigmoid *= certaintyPerSelectedPeer.getValue(peer)
                Pair(familiarClass, sigmoid)
            }.toMap()
            Pair(peer, weightPerFamiliarClass)
        }.toMap()
    }

    private fun weightedAverage(
        weightsPerSelectedPeer: Map<Peer, Map<Class, Weight>>,
        combinedModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean,
        labels: List<Int>,
    ): INDArray {
        val arr = newModel.dup()
        val totalWeights = DoubleArray(newModel.columns()) { 1.0 }
        weightsPerSelectedPeer.forEach { (peer, weights) ->
            weights.forEach { (familiarClass, weight) ->
                val currentClassParameters = arr.getColumn(familiarClass.toLong())
                val extraClassParameters = combinedModels[peer]!!.getColumn(familiarClass.toLong()).mul(weight)
                val newClassParameters = currentClassParameters.add(extraClassParameters)
                arr.putColumn(familiarClass, newClassParameters)
                totalWeights[familiarClass] += weight
            }
        }
        logger.d(true) { "totalWeights: ${totalWeights.contentToString()}" }
        totalWeights.forEachIndexed { index, weight ->
            if (weight != 1.0) {  // optimization
                val scaledColumn = arr.getColumn(index.toLong()).div(weight)
                arr.putColumn(index, scaledColumn/*.subi(newModel.meanNumber())*/)
            }
        }
        return arr
    }

    private fun getCertaintyPerSelectedPeer(
        peers: IntArray,
        peerF1sPerClass: Map<Peer, Map<Class, Double>>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val selectedClasses = selectedClassesPerPeer.getValue(peer)
            val f1PerSelectedClasses = selectedClasses.map { clz -> peerF1sPerClass.getValue(peer).getValue(clz) }.toDoubleArray()
            val average = f1PerSelectedClasses.average()
            val std = f1PerSelectedClasses.std()
            Pair(peer, max(0.0, average - std))
        }.toMap()
    }

    private fun getWeightPerSelectedPeer(
        peers: IntArray,
        unboundedScoresPerSelectedPeer: Map<Peer, Map<Class, Score>>,
        certaintyPerSelectedPeer: Map<Peer, Double>
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val totalScore = unboundedScoresPerSelectedPeer.getValue(peer).values.sum()
            var sigmoid = 1 / (1 + exp(-totalScore / 100))  // sigmoid function
            sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
            sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
            sigmoid = max(0.0, sigmoid)  // no negative weights
            sigmoid *= certaintyPerSelectedPeer.getValue(peer)
            Pair(peer, sigmoid)
        }.toMap()
    }

    private fun incorporateForeignWeights(
        peers: IntArray,
        weightPerSelectedPeer: Map<Peer, Double>,
        combinedModels: Map<Peer, INDArray>,
        weightedAverage: INDArray,
        foreignLabels: List<Int>,
        logging: Boolean
    ): INDArray {
        val arr = weightedAverage.dup()
        logger.d(true) { "foreignLabels: $foreignLabels" }
        peers.forEach { peer ->
            foreignLabels.forEach { label ->
                val targetColumn = combinedModels.getValue(peer).getColumn(label.toLong())
                arr.putColumn(label, arr.getColumn(label.toLong()).add(targetColumn.mul(weightPerSelectedPeer.getValue(peer))))
            }
        }
        val totalWeight = weightPerSelectedPeer.values.sum() + 1   // + 1 for the current node
        logger.d(true) { "totalWeight: $totalWeight" }
        foreignLabels.forEach { label ->
            arr.putColumn(label, arr.getColumn(label.toLong()).div(totalWeight))
        }
        return arr
    }

    private fun clamp(num: Double, min: Double = 0.0, max: Double = 1.0): Double {
        return max(min, min(max, num))
    }

    private fun DoubleArray.std(): Double {
        val avg = this.average()
        val std = this.fold(0.0) { a, b -> a + (b - avg).pow(2) }
        return sqrt(std / this.size)
    }
}
