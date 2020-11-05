package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.IPv4Address
import nl.tudelft.ipv8.Peer
import nl.tudelft.ipv8.keyvault.defaultCryptoProvider
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MessageListener
import nl.tudelft.trustchain.fedml.ipv8.MsgParamUpdate
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.updater.UpdaterCreator
import org.deeplearning4j.nn.workspace.ArrayType
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.memory.enums.AllocationPolicy
import org.nd4j.linalg.api.memory.enums.LearningPolicy
import org.nd4j.linalg.api.memory.enums.ResetPolicy
import org.nd4j.linalg.api.memory.enums.SpillPolicy
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import kotlin.random.Random

private val logger = KotlinLogging.logger("DistributedRunner")

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    private val paramBuffer: MutableList<Pair<INDArray, Int>> = ArrayList()
    private lateinit var random: Random


    /**
     * Workspace for working memory for a single layer: forward pass and backward pass
     * Note that this is opened/closed once per op (activate/backpropGradient call)
     */
    protected val WS_LAYER_WORKING_MEM = "WS_LAYER_WORKING_MEM"

    /**
     * Workspace for storing all layers' activations - used only to store activations (layer inputs) as part of backprop
     * Not used for inference
     */
    protected val WS_ALL_LAYERS_ACT = "WS_ALL_LAYERS_ACT"

    /**
     * Next 2 workspaces: used for:
     * (a) Inference: holds activations for one layer only
     * (b) Backprop: holds activation gradients for one layer only
     * In both cases, they are opened and closed on every second layer
     */
    protected val WS_LAYER_ACT_1 = "WS_LAYER_ACT_1"
    protected val WS_LAYER_ACT_2 = "WS_LAYER_ACT_2"

    /**
     * Workspace for output methods that use OutputAdapter
     */
    protected val WS_OUTPUT_MEM = "WS_OUTPUT_MEM"

    /**
     * Workspace for working memory in RNNs - opened and closed once per RNN time step
     */
    protected val WS_RNN_LOOP_WORKING_MEM = "WS_RNN_LOOP_WORKING_MEM"


    protected var WS_LAYER_WORKING_MEM_CONFIG: WorkspaceConfiguration? = null

    protected val WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
        .initialSize(0)
        .overallocationLimit(0.05)
        .policyLearning(LearningPolicy.FIRST_LOOP)
        .policyReset(ResetPolicy.BLOCK_LEFT)
        .policySpill(SpillPolicy.REALLOCATE)
        .policyAllocation(AllocationPolicy.OVERALLOCATE)
        .build()

    protected var WS_LAYER_ACT_X_CONFIG: WorkspaceConfiguration? = null

    protected val WS_RNN_LOOP_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
        .initialSize(0).overallocationLimit(0.05).policyReset(ResetPolicy.BLOCK_LEFT)
        .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
        .policyLearning(LearningPolicy.FIRST_LOOP).build()


    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        val nodeAssignments = getNodeAssignments(baseDirectory)
        val port = community.myEstimatedWan.port
        val behavior = nodeAssignments?.get(port) ?: Behaviors.BENIGN
        val otherNodes = nodeAssignments?.keys?.filter { it != port } ?: arrayListOf()
        val otherPeers = convertToPeer(otherNodes)
        this.random = Random(seed)
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.datasetIteratorConfiguration,
                mlConfiguration.trainConfiguration.behavior,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.datasetIteratorConfiguration,
                mlConfiguration.trainConfiguration.behavior,
                seed
            )
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "distributed",
                mlConfiguration,
                seed,
                listOf(
                    "before or after averaging",
                    "total samples",
                    "#peers included in current batch"
                )
            )
            val network = generateNetwork(
                mlConfiguration.dataset,
                mlConfiguration.nnConfiguration,
                seed
            )
            network.setListeners(ScoreIterationListener(printScoreIterations))

            trainTestSendNetwork(
                network,
                evaluationProcessor,
                trainDataSetIterator,
                testDataSetIterator,
                mlConfiguration.trainConfiguration,
                behavior,
                otherPeers
            )
        }
    }

    private fun convertToPeer(otherNodes: List<Int>): List<Peer> {
        val key = defaultCryptoProvider.generateKey()
        return otherNodes.map { Peer(key, IPv4Address("10.0.2.2", it), supportsUTP = true) }
    }

    private fun getNodeAssignments(baseDirectory: File): Map<Int, Behaviors>? {
        val fileNodeAssignment = File(baseDirectory, "NodeAssignment.csv")

        if (!fileNodeAssignment.exists()) {
            return null
        }

        val lines = fileNodeAssignment.readLines()
        val split = lines.map { it.split(",") }
        return split.associateBy({ it[0].toInt() }, { spl -> Behaviors.values().first { it.id == spl[1] } })
    }

    private fun trainTestSendNetwork(
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        trainDataSetIterator: DataSetIterator,
        testDataSetIterator: DataSetIterator,
        trainConfiguration: TrainConfiguration,
        behavior: Behaviors,
        otherPeers: List<Peer>
    ) {
        print(behavior)
        val batchSize = trainDataSetIterator.batch()
        val gar = trainConfiguration.gar.obj
        var samplesCounter = 0
        var epoch = 0
        var iterations = 0
        var iterationsToEvaluation = 0
        for (i in 0 until trainConfiguration.numEpochs.value) {
            trainDataSetIterator.reset()
            logger.debug { "Starting epoch: $epoch" }
            evaluationProcessor.epoch = epoch
            val start = System.currentTimeMillis()
            while (true) {

                // Train
                var endEpoch = false
                try {
//                    network.fit(trainDataSetIterator.next())
                    val batch = trainDataSetIterator.next()
                    val pair = network.calculateGradients(batch.features, batch.labels, null, null)
//                    network.computeGradientAndScore()
//                    val gradAndScore = network.gradientAndScore()

                    val updater = UpdaterCreator.getUpdater(network)
                    updater.update(network, pair.first, iterations, epoch, network.batchSize(), LayerWorkspaceMgr.noWorkspaces())
//                    val gradient = network.gradient()
//                    network.update(gradient)
                    val params = network.params()
                    params.subi(pair.first.gradient())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                samplesCounter += batchSize
                iterations += batchSize
                iterationsToEvaluation += batchSize

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {

                    // Test
                    iterationsToEvaluation = 0
                    val end = System.currentTimeMillis()
                    logger.debug { "Evaluating network " }
                    evaluationProcessor.iteration = iterations
                    execEvaluationProcessor(
                        evaluationProcessor,
                        testDataSetIterator,
                        network,
                        EvaluationProcessor.EvaluationData(
                            "before", samplesCounter, "", end - start, network.iterationCount, epoch
                        )
                    )

                    // Integrate parameters of other peers
                    var ret = -1
                    val numPeers = paramBuffer.size + 1
                    val averageParams: Pair<INDArray, Int>
                    if (numPeers == 1) {
                        logger.debug { "No received params => skipping integration evaluation" }
                        evaluationProcessor.skip()
                        averageParams = Pair(network.params().dup(), samplesCounter)
                    } else {
                        logger.debug { "Params received => executing aggregation rule" }

                        val start2 = System.currentTimeMillis()
                        val model = Pair(network.params().dup(), samplesCounter)
                        averageParams = gar.integrateParameters(model, paramBuffer, network, testDataSetIterator)
                        ret = averageParams.second
                        paramBuffer.clear()
                        network.setParameters(averageParams.first)
                        val end2 = System.currentTimeMillis()

                        execEvaluationProcessor(
                            evaluationProcessor,
                            testDataSetIterator,
                            network,
                            EvaluationProcessor.EvaluationData(
                                "after", samplesCounter, numPeers.toString(), end2 - start2, iterations, epoch
                            )
                        )
                    }

                    // Send new parameters to other peers
                    val sendMessage = when (trainConfiguration.communicationPattern) {
                        CommunicationPatterns.ALL -> community::sendToAll
                        CommunicationPatterns.RANDOM -> community::sendToRandomPeer
                        CommunicationPatterns.RR -> community::sendToNextPeerRR
                    }
                    val message = craftMessage(averageParams.first, trainConfiguration.behavior/*behavior*/)
                    sendMessage(MessageId.MSG_PARAM_UPDATE, MsgParamUpdate(message, samplesCounter), otherPeers)
                    val newSamplesCounter = ret
                    samplesCounter = if (newSamplesCounter == -1) samplesCounter else newSamplesCounter
                }
                if (endEpoch) {
                    break
                }
            }
            epoch++
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
    }

    private fun craftMessage(first: INDArray, behavior: Behaviors): INDArray {
        return when (behavior) {
            Behaviors.BENIGN -> first
            Behaviors.NOISE -> craftNoiseMessage(first)
            Behaviors.LABEL_FLIP -> first
        }
    }

    private fun craftNoiseMessage(first: INDArray): INDArray {
        val oldMatrix = first.toFloatMatrix()[0]
        val newMatrix = Array(1) { FloatArray(oldMatrix.size) }
        for (i in oldMatrix.indices) {
            newMatrix[0][i] = random.nextFloat() * 2 - 1
        }
        return NDArray(newMatrix)
    }

    private fun execEvaluationProcessor(
        evaluationProcessor: EvaluationProcessor,
        testDataSetIterator: DataSetIterator,
        network: MultiLayerNetwork,
        evaluationData: EvaluationProcessor.EvaluationData
    ) {
        testDataSetIterator.reset()
        evaluationProcessor.extraElements = mapOf(
            Pair("before or after averaging", evaluationData.beforeAfterAveraging),
            Pair("total samples", evaluationData.samplesCounter.toString()),
            Pair("#peers included in current batch", evaluationData.numPeers)
        )
        evaluationProcessor.elapsedTime = evaluationData.elapsedTime
        val evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
        evaluationListener.callback = evaluationProcessor
        evaluationListener.iterationDone(
            network,
            evaluationData.iterationCount,
            evaluationData.epoch
        )
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        logger.debug { "onMessageReceived" }
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                val paramUpdate = payload as MsgParamUpdate
                paramBuffer.add(Pair(paramUpdate.array, paramUpdate.weight))
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }
}
