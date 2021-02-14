package nl.tudelft.trustchain.fedml.ai.gar

import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.math.*

private const val NUM_MODELS_EXPLOITATION = 50
private const val NUM_MODELS_EXPLORATION = 5
private const val MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING = 20

private typealias Peer = Int
private typealias Class = Int
private typealias SeqAttackPenalty = Double
private typealias Score = Double
private typealias Weight = Double
private typealias Certainty = Double

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
        debug(logging) { formatName("BRISTLE") }
        debug(logging) { "Found ${newOtherModels.size} other models" }
        debug(logging) { "oldModel: ${oldModel.getFloat(0)}" }
        val newModel = oldModel.sub(gradient)
        debug(logging) { "newModel: ${newModel.getFloat(0)}" }
        debug(logging) { "otherModels: ${newOtherModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val distances = getDistances(newModel, newOtherModels, recentOtherModels, logging)
        debug(logging) { "distances: $distances" }
        val exploitationModels = distances
            .keys
            .take(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .toMap()
        debug(logging) { "closeModels: ${exploitationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val explorationModels = distances
            .keys
            .drop(NUM_MODELS_EXPLOITATION)
            .filter { it < 1000000 }
            .map { Pair(it, newOtherModels.getValue(it)) }
            .shuffled()
            .take(NUM_MODELS_EXPLORATION)
            .toMap()
        debug(logging) { "notCloseModels: ${explorationModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }

        val combinedModels = HashMap<Peer, INDArray>()
        combinedModels.putAll(exploitationModels)
        combinedModels.putAll(explorationModels)
        debug(logging) { "combinedModels: ${combinedModels.map { it.value.getFloat(0) }.toCollection(ArrayList())}" }
        val peers = combinedModels.keys.sorted().toIntArray()
        testDataSetIterator.reset()

        val myRecalls = calculateRecallPerClass(newModel, network, testDataSetIterator)
        debug(logging) { "myRecallPerClass: ${myRecalls.toList()}" }

        val recallsPerPeer = calculateRecallPerClass(combinedModels, network, testDataSetIterator)
        debug(logging) { "peerRecallPerClass: ${recallsPerPeer.map { Pair(it.key, it.value.toList()) }}}" }

        val selectedClassesPerPeer = getBestPerformingClassesPerPeer(peers, myRecalls, recallsPerPeer, countPerPeer)
        debug(logging) { "selectedClassesPerPeer: ${selectedClassesPerPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedDiffsPerSelectedPeer = getWeightedDiffsPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer)
        debug(logging) { "weightedDiffPerSelectedPeer: ${weightedDiffsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val seqAttackPenaltiesPerSelectedPeer = getSeqAttackPenaltiesPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer)
        debug(logging) { "seqAttackPenaltyPerSelectedPeer: ${seqAttackPenaltiesPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val scoresPerSelectedPeer = getScoresPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer, weightedDiffsPerSelectedPeer, seqAttackPenaltiesPerSelectedPeer, true)
        debug(logging) { "scoresPerSelectedPeer: ${scoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val certaintiesPerSelectedPeer = getCertaintiesPerSelectedPeer(peers, selectedClassesPerPeer, recallsPerPeer)
        debug(logging) { "certaintiesPerSelectedPeer: ${certaintiesPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightsPerSelectedPeer = getWeightsPerSelectedPeer(peers, selectedClassesPerPeer, scoresPerSelectedPeer, certaintiesPerSelectedPeer)
        debug(logging) { "weightsPerSelectedPeer: ${weightsPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val weightedAverage = weightedAverage(weightsPerSelectedPeer, combinedModels, newModel, logging)
        debug(logging) { "weightedAverage: $weightedAverage" }


        val unboundedScoresPerSelectedPeer = getScoresPerSelectedPeer(peers, myRecalls, recallsPerPeer, selectedClassesPerPeer, weightedDiffsPerSelectedPeer, seqAttackPenaltiesPerSelectedPeer, false)
        debug(logging) { "unboundedScoresPerSelectedPeer: ${unboundedScoresPerSelectedPeer.map { Pair(it.key, it.value.toList()) }}" }

        val certaintyPerSelectedPeer = getCertaintyPerSelectedPeer(peers, recallsPerPeer)
        debug(logging) { "certaintyPerSelectedPeer: $certaintyPerSelectedPeer" }

        val weightPerSelectedPeer = getWeightPerSelectedPeer(peers, unboundedScoresPerSelectedPeer, certaintyPerSelectedPeer)
        debug(logging) { "weightPerSelectedPeer: $weightPerSelectedPeer" }

        val result = incorporateForeignWeights(peers, weightPerSelectedPeer, combinedModels, weightedAverage, testDataSetIterator.labels, logging)
        debug(logging) { "result: $result" }
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
            debug(logging) { "Distance calculated: $min" }
            distances[index] = min
        }
        for (i in 0 until min(MIN_NUMBER_MODELS_FOR_DISTANCE_SCREENING - distances.size, recentOtherModels.size)) {
            val otherModel = recentOtherModels.elementAt(recentOtherModels.size - 1 - i)
            distances[1100000 + otherModel.first] = otherModel.second.distance2(newModel)
        }
        return distances.toList().sortedBy { (_, value) -> value }.toMap()
    }

    private fun calculateRecallPerClass(
        model: INDArray,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
    ): DoubleArray {
        network.outputLayer.setParam("W", model)
        val evaluations = arrayOf(Evaluation())
        network.doEvaluation(testDataSetIterator, *evaluations)
        return testDataSetIterator.labels.map {
            evaluations[0].recall(it.toInt())
        }.toDoubleArray()
    }

    private fun calculateRecallPerClass(
        models: Map<Int, INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: CustomDataSetIterator,
    ): Map<Peer, DoubleArray> {
        return models.map { (index, model) ->
            Pair(index, calculateRecallPerClass(model, network, testDataSetIterator))
        }.toMap()
    }

    private fun getBestPerformingClassesPerPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        countPerPeer: Map<Int, Int>
    ): Map<Peer, IntArray> {
        return peers.map { peer ->
            val selectionSize = countPerPeer.getOrDefault(peer, myRecallPerClass.size)
            val selectedClasses = if (selectionSize == peerRecallPerClass.getValue(peer).size)
                (0 until selectionSize).toList().toIntArray()
            else
                peerRecallPerClass.getValue(peer)
                    .mapIndexed { label, recall -> Pair(label, recall) }
                    .sortedByDescending { it.second }
                    .take(selectionSize)  // Attackers have a negative index => assume that they have the same classes as the current peer to maximize attack strength
                    .map { it.first }
                    .toIntArray()
            Pair(peer, selectedClasses)
        }.toMap()
    }

    private fun getWeightedDiffsPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>
    ): Map<Peer, Map<Class, Double>> {
        return peers.map { peer ->
            val weightDiffsPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                Pair(selectedClass, abs(myRecallPerClass[selectedClass] - peerRecallPerClass.getValue(peer)[selectedClass]) * 10)
            }.toMap()
            Pair(peer, weightDiffsPerSelectedClass)
        }.toMap()
    }

    private fun getSeqAttackPenaltiesPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>
    ): Map<Peer, Map<Class, SeqAttackPenalty>> {
        return peers.map { peer ->
            val seqAttackPenaltiesPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                var seqAttackPenalty = 0.0
                for (i in 0 until peer) {
                    seqAttackPenalty += max(0.0, 2 * (myRecallPerClass[selectedClass] - peerRecallPerClass.getValue(peer)[selectedClass]))
                }
                Pair(selectedClass, seqAttackPenalty)
            }.toMap()
            Pair(peer, seqAttackPenaltiesPerSelectedClass)
        }.toMap()
    }

    private fun getScoresPerSelectedPeer(
        peers: IntArray,
        myRecallPerClass: DoubleArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        weightedDiffPerSelectedPeer: Map<Peer, Map<Class, Double>>,
        seqAttackPenaltyPerSelectedPeer: Map<Peer, Map<Class, SeqAttackPenalty>>,
        bounded: Boolean
    ): Map<Peer, Map<Class, Score>> {
        return peers.map { peer ->
            val scorePerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                val myRecall = myRecallPerClass[selectedClass]
                val peerRecall = peerRecallPerClass.getValue(peer)[selectedClass]
                val weightedDiff = weightedDiffPerSelectedPeer.getValue(peer).getValue(selectedClass)
                val seqAttackPenalty = seqAttackPenaltyPerSelectedPeer.getValue(peer).getValue(selectedClass)
                val score = if (peerRecall > myRecall) weightedDiff.pow(3 + myRecallPerClass[selectedClass]) else -1.0 * weightedDiff.pow(4 + myRecall) * (1 + seqAttackPenalty)
                Pair(selectedClass, if (bounded) max(-100.0, score) else score)
            }.toMap()
            Pair(peer, scorePerSelectedClass)
        }.toMap()
    }

    private fun getCertaintiesPerSelectedPeer(
        peers: IntArray,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        recallsPerPeer: Map<Peer, DoubleArray>
    ): Map<Peer, Map<Class, Certainty>> {
        return peers.map { peer ->
            val weightPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                val certainty = clamp((recallsPerPeer.getValue(peer)[selectedClass] - 0.2) * 4)
                Pair(selectedClass, certainty)
            }.toMap()
            Pair(peer, weightPerSelectedClass)
        }.toMap()
    }

    private fun getWeightsPerSelectedPeer(
        peers: IntArray,
        selectedClassesPerPeer: Map<Peer, IntArray>,
        scoresPerSelectedPeer: Map<Peer, Map<Class, Score>>,
        certaintiesPerSelectedPeer: Map<Peer, Map<Class, Certainty>>
    ): Map<Peer, Map<Class, Weight>> {
        return peers.map { peer ->
            val weightPerSelectedClass = selectedClassesPerPeer.getValue(peer).map { selectedClass ->
                var sigmoid = 1 / (1 + exp(-scoresPerSelectedPeer.getValue(peer).getValue(selectedClass) / 100))  // sigmoid function
                sigmoid *= 10  // sigmoid from 0 -> 1 to 0 -> 10
                sigmoid -= 4  // sigmoid from 0 -> 10 to -4 -> 6
                sigmoid = max(0.0, sigmoid)  // no negative weights
                sigmoid *= certaintiesPerSelectedPeer.getValue(peer).getValue(selectedClass)
                Pair(selectedClass, sigmoid)
            }.toMap()
            Pair(peer, weightPerSelectedClass)
        }.toMap()
    }

    private fun weightedAverage(
        weightsPerSelectedPeer: Map<Peer, Map<Class, Weight>>,
        combinedModels: Map<Int, INDArray>,
        newModel: INDArray,
        logging: Boolean,
    ): INDArray {
        val arr = newModel.dup()
        val totalWeights = DoubleArray(newModel.columns()) { 1.0 }
        weightsPerSelectedPeer.forEach { (peer, weights) ->
            weights.forEach { (clazz, weight) ->
                arr.putColumn(clazz, combinedModels[peer]!!.getColumn(clazz.toLong()).mul(weight))
                totalWeights[clazz] += weight
            }
        }
        debug(logging) { "totalWeights: ${totalWeights.contentToString()}" }
        totalWeights.forEachIndexed { index, weight ->
            arr.putColumn(index, arr.getColumn(index.toLong()).div(weight))
        }
        return arr
    }

    private fun getCertaintyPerSelectedPeer(
        peers: IntArray,
        peerRecallPerClass: Map<Peer, DoubleArray>,
    ): Map<Peer, Double> {
        return peers.map { peer ->
            val average = peerRecallPerClass.getValue(peer).average()
            val std = peerRecallPerClass.getValue(peer).std()
            Pair(peer, max(0.0, average - std * 2))
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
        labels: List<String>,
        logging: Boolean
    ): INDArray {
        val arr = weightedAverage.dup()
        val foreignLabels = (0 until weightedAverage.columns()).subtract(labels.map { it.toInt() })
        debug(logging) { "foreignLabels: $foreignLabels" }
        peers.forEach { peer ->
            foreignLabels.forEach { label ->
                val targetColumn = combinedModels.getValue(peer).getColumn(label.toLong())
                arr.putColumn(label, targetColumn.mul(weightPerSelectedPeer.getValue(peer)))
            }
        }
        val totalWeight = weightPerSelectedPeer.values.sum() + 1   // + 1 for the current node
        debug(logging) { "totalWeight: $totalWeight" }
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
        debug(true) { "  => avg: $avg" }
        val std = this.fold(0.0) { a, b -> a + (b - avg).pow(2) }
        debug(true) { "  => std: $std" }
        debug(true) { "  => avg: $this" }
        debug(true) { "  => size: $size" }
        return sqrt(std / this.size)
    }
}
