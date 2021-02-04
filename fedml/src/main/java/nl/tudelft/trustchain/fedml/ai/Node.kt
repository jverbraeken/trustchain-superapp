package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import java.io.File
import java.math.BigInteger
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.round
import kotlin.random.Random

private val bigPrime = BigInteger("100012421")
private val logger = KotlinLogging.logger("Node")
private const val ONLY_EVALUATE_FIRST_NODE = true
private const val SIZE_RECENT_OTHER_MODELS = 20

class Node(
    private val nodeIndex: Int,
    testConfig: MLConfiguration,
    private val generateNetwork: (architecture: (nnConfiguration: NNConfiguration, seed: Int) -> MultiLayerConfiguration, nnConfiguration: NNConfiguration, seed: Int) -> MultiLayerNetwork,
    getDataSetIterators: (inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDataSetType, baseDirectory: File, behavior: Behaviors) -> CustomDataSetIterator, datasetIteratorConfiguration: DatasetIteratorConfiguration, seed: Long, baseDirectory: File, behavior: Behaviors) -> List<CustomDataSetIterator>,
    baseDirectory: File,
    private val evaluationProcessor: EvaluationProcessor,
    private val start: Long,
    val shareModel: (
        params: INDArray,
        trainConfiguration: TrainConfiguration,
        random: Random,
        nodeIndex: Int,
        countPerPeer: Map<Int, Int>
    ) -> Unit
) {
    private val dataset = testConfig.dataset
    private val recentOtherModelsBuffer = ArrayDeque<Pair<Int, INDArray>>()
    private val newOtherModelBufferTemp = ConcurrentHashMap<Int, INDArray>()
    private val newOtherModelBuffer = ConcurrentHashMap<Int, INDArray>()
    private val random = Random(nodeIndex)
    private val sraKeyPair = SRAKeyPair.create(bigPrime, java.util.Random(nodeIndex.toLong()))

    private var network: MultiLayerNetwork

    private val datasetIteratorConfiguration: DatasetIteratorConfiguration
    private val distribution: List<Int>
    private val usedClassIndices: List<Int>

    private val nnConfiguration: NNConfiguration

    private val trainConfiguration: TrainConfiguration
    private val behavior: Behaviors
    private val iterationsBeforeEvaluation: Int
    private val iterationsBeforeSending: Int
    private var joiningLateRemainingIterations: Int
    private val slowdown: Slowdowns
    private val gar: GARs

    private val modelPoisoningConfiguration: ModelPoisoningConfiguration
    private val modelPoisoningAttack: ModelPoisoningAttacks
    private val numAttackers: NumAttackers

    private var oldParams: INDArray
    private var newParams: INDArray
    private var gradient: INDArray
    private val iterTrain: CustomDataSetIterator
    private val iterTest: CustomDataSetIterator
    private val iterTestFull: CustomDataSetIterator
    private val logging: Boolean

    private lateinit var cw: INDArray
    private lateinit var countPerPeer: Map<Int, Int>
    private var slowdownRemainingIterations = 0

    init {

        network = generateNetwork(dataset.architecture, testConfig.nnConfiguration, nodeIndex)

        datasetIteratorConfiguration = testConfig.datasetIteratorConfiguration
        distribution = datasetIteratorConfiguration.distribution
        usedClassIndices = distribution.mapIndexed { ind, v -> if (v > 0) ind else null }.filterNotNull()

        nnConfiguration = testConfig.nnConfiguration

        trainConfiguration = testConfig.trainConfiguration
        behavior = trainConfiguration.behavior
        iterationsBeforeEvaluation = trainConfiguration.iterationsBeforeEvaluation!!
        iterationsBeforeSending = trainConfiguration.iterationsBeforeSending!!
        joiningLateRemainingIterations = trainConfiguration.joiningLate.rounds * iterationsBeforeSending
        slowdown = trainConfiguration.slowdown
        gar = trainConfiguration.gar

        modelPoisoningConfiguration = testConfig.modelPoisoningConfiguration
        modelPoisoningAttack = modelPoisoningConfiguration.attack
        numAttackers = modelPoisoningConfiguration.numAttackers

        oldParams = if (gar == GARs.BRISTLE) network.outputLayer.params().dup() else network.params().dup()
        newParams = NDArray()
        gradient = NDArray()
        val iters = getDataSetIterators(
            dataset.inst,
            datasetIteratorConfiguration,
            nodeIndex.toLong() * 10,
            baseDirectory,
            behavior
        )
        iterTrain = iters[0]
        iterTest = iters[1]
        iterTestFull = iters[2]

        logging = nodeIndex == 0 || !ONLY_EVALUATE_FIRST_NODE
    }

    fun pretrainNetwork(iterations: Int, start: Long) {
        repeat(iterations) { iteration ->
            val elem = try {
                iterTrain.next()
            } catch (e: NoSuchElementException) {
                iterTrain.reset()
                iterTrain.next()
            }
            network.fit(elem)

            if (iteration % iterationsBeforeEvaluation == 0) {
                val elapsedTime2 = System.currentTimeMillis() - start
                val extraElements2 = mapOf(
                    Pair("before or after averaging", "before"),
                    Pair("#peers included in current batch", "1")
                )
                evaluationProcessor.evaluate(
                    iterTestFull,
                    network,
                    extraElements2,
                    elapsedTime2,
                    iteration,
                    0,
                    0,
                    true
                )
            }
        }
        val initParams = network.params().dup()
        reInitializeWithFrozen(initParams)
    }

    fun reInitializeWithFrozen(preTrainedNetwork: INDArray) {
        network = generateNetwork(::generateDefaultMNISTConfigurationFrozen, nnConfiguration, nodeIndex)
        network.setParams(preTrainedNetwork.dup())
        cw = network.outputLayer.params().dup()
        cw.muli(0)
        updateCW()
    }

    fun performIteration(epoch: Int, iteration: Int): Boolean {
//        logger.debug { "Node: $nodeIndex" }

        newParams = if (gar == GARs.BRISTLE) network.outputLayer.params().dup() else network.params().dup()
        gradient = oldParams.sub(newParams)

        if (joiningLateSkip()) {
            return false
        }
        if (slowdownSkip()) {
            return false
        }

        if (iteration % iterationsBeforeSending == 0) {
            if (behavior == Behaviors.BENIGN) {
                addPotentialAttacks()
                potentiallyIntegrateParameters(iteration)
                potentiallyEvaluate(epoch, iteration, "before")
            }
            newOtherModelBuffer.clear()
        }

        if (gar == GARs.BRISTLE) {
            resetTW()
        }

        oldParams = if (gar == GARs.BRISTLE) network.outputLayer.params().dup() else network.params().dup()

        val epochEnd = fitNetwork(network, iterTrain)

        if (gar == GARs.BRISTLE) {
            updateCW()
        }

        if (iteration % iterationsBeforeSending == 0) {
            shareModel(
                if (gar == GARs.BRISTLE) network.outputLayer.params().dup() else network.params().dup(),
                trainConfiguration,
                random,
                nodeIndex,
                countPerPeer
            )
        }

        potentiallyEvaluate(epoch, iteration, "after")
        return epochEnd
    }

    private fun updateCW() {
        val tw = network.outputLayer.params()
        for (index in usedClassIndices) {
            cw.putColumn(index, tw.getColumn(index.toLong()))
        }
    }

    private fun resetTW() {
        val tw = network.outputLayer.params()
        tw.muli(0)
        for (index in usedClassIndices) {
            tw.putColumn(index, cw.getColumn(index.toLong()))
        }
        network.outputLayer.setParams(tw)
    }

    private fun joiningLateSkip(): Boolean {
        if (joiningLateRemainingIterations > 0) {
            joiningLateRemainingIterations--
            if (nodeIndex == 0) logger.debug { "JL => continue" }
            newOtherModelBuffer.clear()
            return true
        }
        return false
    }

    private fun slowdownSkip(): Boolean {
        if (slowdown != Slowdowns.NONE) {
            if (slowdownRemainingIterations > 0) {
                slowdownRemainingIterations--
                if (nodeIndex == 0) logger.debug { "SD => continue" }
                newOtherModelBuffer.clear()
                return true
            } else {
                slowdownRemainingIterations = round(1 / slowdown.multiplier).toInt() - 1
            }
        }
        return false
    }

    private fun fitNetwork(network: MultiLayerNetwork, dataSetIterator: CustomDataSetIterator): Boolean {
        try {
            val ds = dataSetIterator.next()
            network.fit(ds)
        } catch (e: Exception) {
            dataSetIterator.reset()
            return true
        }
        return false
    }

    private fun addPotentialAttacks() {
        val attackVectors = modelPoisoningAttack.obj.generateAttack(
            numAttackers,
            oldParams,
            gradient,
            newOtherModelBuffer,
            random
        )
        newOtherModelBuffer.putAll(attackVectors)
    }

    private fun potentiallyIntegrateParameters(iteration: Int) {
        val numPeers = newOtherModelBuffer.size + 1
        if (numPeers > 1) {
            val averageParams = gar.obj.integrateParameters(
                network,
                oldParams,
                gradient,
                newOtherModelBuffer,
                recentOtherModelsBuffer,
                iterTest,
                countPerPeer,
                logging && (iteration % iterationsBeforeEvaluation == 0)
            )
            if (gar == GARs.BRISTLE) {
                for (index in 0 until cw.columns()) {
                    cw.putColumn(index, averageParams.getColumn(index.toLong()))
                }
            } else {
                network.setParameters(averageParams)
            }
            recentOtherModelsBuffer.addAll(newOtherModelBuffer.toList())
            while (recentOtherModelsBuffer.size > SIZE_RECENT_OTHER_MODELS) {
                recentOtherModelsBuffer.removeFirst()
            }
        }
    }

    private fun potentiallyEvaluate(epoch: Int, iteration: Int, beforeOrAfter: String) {
        if (logging && (iteration % iterationsBeforeEvaluation == 0)) {
            val evaluationScript = {
                val elapsedTime2 = System.currentTimeMillis() - start
                val extraElements2 = mapOf(
                    Pair("before or after averaging", beforeOrAfter),
                    Pair("#peers included in current batch", newOtherModelBuffer.size.toString())
                )
                evaluationProcessor.evaluate(
                    iterTestFull,
                    network,
                    extraElements2,
                    elapsedTime2,
                    iteration,
                    epoch,
                    nodeIndex,
                    nodeIndex == 0
                )
            }
            if (gar == GARs.BRISTLE) {
                val oldTw = network.outputLayer.params().dup()
                network.outputLayer.setParams(cw)

                evaluationScript.invoke()

                network.outputLayer.setParams(oldTw)
            } else {
                evaluationScript.invoke()
            }
        }
    }

    fun applyNetworkBuffers() {
        newOtherModelBuffer.putAll(newOtherModelBufferTemp)
        newOtherModelBufferTemp.clear()
    }

    fun getNodeIndex(): Int {
        return nodeIndex
    }

    fun addNetworkMessage(from: Int, message: INDArray) {
        newOtherModelBufferTemp[from] = message
    }

    fun getSRAKeyPair(): SRAKeyPair {
        return sraKeyPair
    }

    fun printIterations() {
        network.setListeners(ScoreIterationListener(5))
    }

    fun getNetworkParams(): INDArray {
        return network.params().dup()
    }

    fun getLabels(): List<String> {
        return iterTrain.labels
    }

    fun setCountPerPeer(countPerPeer: Map<Int, Int>) {
        this.countPerPeer = countPerPeer
    }
}
