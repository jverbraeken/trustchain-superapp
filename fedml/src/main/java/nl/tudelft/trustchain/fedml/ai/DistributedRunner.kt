package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ipv8.*
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.random.Random
import kotlin.collections.ArrayDeque

private val logger = KotlinLogging.logger("DistributedRunner")
private const val SIZE_RECENT_OTHER_MODELS = 20
private val psiRequests = ConcurrentHashMap.newKeySet<Int>()

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    internal lateinit var baseDirectory: File
    private val newOtherModels = ConcurrentHashMap<Int, MsgParamUpdate>()
    private val recentOtherModelsBuffer = ArrayDeque<Pair<Int, INDArray>>()
    private val psiCaMessagesFromServers = CopyOnWriteArrayList<MsgPsiCaServerToClient>()
    private lateinit var random: Random
    private lateinit var labels: List<String>
    private lateinit var sraKeyPair: SRAKeyPair
    private var deferred = CompletableDeferred<Unit>()

    init {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        FedMLCommunity.registerMessageListener(MessageId.MSG_PSI_CA_CLIENT_TO_SERVER, this)
        FedMLCommunity.registerMessageListener(MessageId.MSG_PSI_CA_SERVER_TO_CLIENT, this)
        FedMLCommunity.registerMessageListener(MessageId.MSG_NEW_TEST_COMMAND, this)
    }

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration,
    ) {
    }

    private suspend fun getSimilarPeers(
        port: Int,
        labels: List<String>,
        sraKeyPair: SRAKeyPair,
        numPeers: Int,
        psiCaMessagesFromServers: CopyOnWriteArrayList<MsgPsiCaServerToClient>,
    ): Map<Int, Int> {
        val encryptedLabels = clientsRequestsServerLabels(
            labels,
            sraKeyPair
        )
        community.sendToAll(
            MessageId.MSG_PSI_CA_CLIENT_TO_SERVER,
            MsgPsiCaClientToServer(encryptedLabels, port),
            reliable = true
        )

        deferred.complete(Unit)

        var count = 0
        while (psiCaMessagesFromServers.size < numPeers) {
            delay(200)
            count += 200
            logger.debug { "PSI_CA found ${psiCaMessagesFromServers.size} out of $numPeers" }
        }

        return clientReceivesServerResponses(
            port,
            psiCaMessagesFromServers,
            sraKeyPair
        )
    }

    private suspend fun trainTestSendNetwork(
        // General information
        network: MultiLayerNetwork,

        // Training the network
        iterTrain: CustomDataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Evaluation results
        evaluationProcessor: EvaluationProcessor,
        iterTestFull: CustomDataSetIterator,

        // Integrating and distributing information to peers
        iterTest: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        usedClassIndices: List<Int>,
    ) {
        /**
         * Joining late logic...
         */

        val gar = trainConfiguration.gar
        var epoch = 0
        val start = System.currentTimeMillis()
        val cw = network.outputLayer.paramTable().getValue("W").dup()

        var oldParams = if (gar == GARs.BRISTLE) network.outputLayer.paramTable().getValue("W").dup() else network.params().dup()
        var newParams: INDArray
        var gradient: INDArray
        val udpEndpoint = community.endpoint.udpEndpoint!!
        val joiningLateRounds = trainConfiguration.joiningLate.rounds
        val slowdown = ((1.0 / trainConfiguration.slowdown.multiplier) - 1).toInt()

        val iterationsBeforeSending = trainConfiguration.iterationsBeforeSending
        val iterationsBeforeEvaluation = trainConfiguration.iterationsBeforeEvaluation
        val behavior = trainConfiguration.behavior
        val communicationPattern = trainConfiguration.communicationPattern
        val modelPoisoningAttack = modelPoisoningConfiguration.attack
        val numAttackers = modelPoisoningConfiguration.numAttackers

//        TODO slowdown and joiningLateRounds
        for (iteration in 0 until trainConfiguration.maxIteration.value) {
            logger.debug { "Iteration: $iteration" }
            newParams = if (gar == GARs.BRISTLE) network.outputLayer.paramTable().getValue("W").dup() else network.params().dup()
            gradient = oldParams.sub(newParams)

            val newOtherModelBuffer = newOtherModels.map { Pair(it.key, it.value.array) }.toMap().toMutableMap()
            // ADD ATTACKS
            if (iteration % iterationsBeforeSending == 0) {
                val attackVectors = modelPoisoningAttack.obj.generateAttack(
                    numAttackers,
                    oldParams,
                    gradient,
                    newOtherModelBuffer,
                    random
                ).map { Pair(it.key, it.value) }
                newOtherModelBuffer.putAll(attackVectors)
            }

            // INTEGRATE PARAMETERS
            val numPeers = newOtherModels.size + 1
            if (numPeers > 1) {
                logger.debug { "Params received => executing aggregation rule" }
                val averageParams = gar.obj.integrateParameters(
                    network,
                    oldParams,
                    gradient,
                    newOtherModelBuffer,
                    recentOtherModelsBuffer,
                    iterTest,
                    countPerPeer,
                    true
                )
                if (behavior == Behaviors.BENIGN) {
                    if (gar == GARs.BRISTLE) {
                        for (index in 0 until cw.columns()) {
                            cw.putColumn(index, averageParams.getColumn(index.toLong()).dup())
                        }
                        for (index in 0 until cw.columns()) {
                            cw.putColumn(index, cw.getColumn(index.toLong()).sub(cw.getColumn(index.toLong()).meanNumber()))
                        }
                    } else {
                        network.setParameters(averageParams)
                    }
                }
                recentOtherModelsBuffer.addAll(newOtherModelBuffer.toList())
                while (recentOtherModelsBuffer.size > SIZE_RECENT_OTHER_MODELS) {
                    recentOtherModelsBuffer.removeFirst()
                }
            } else {
                logger.debug { "No received params => skipping integration evaluation" }
            }

            // EVALUATION BEFORE
            if (iteration % iterationsBeforeEvaluation == 0) {
                val evaluationScript = {
                    val elapsedTime2 = System.currentTimeMillis() - start
                    val extraElements2 = mapOf(
                        Pair("before or after averaging", "before"),
                        Pair("#peers included in current batch", newOtherModelBuffer.size.toString())
                    )
                    sendEvaluationToMaster(
                        evaluationProcessor.evaluate(
                            iterTestFull,
                            network,
                            extraElements2,
                            elapsedTime2,
                            iteration,
                            epoch,
                            0,
                            true
                        )
                    )
                }
                if (gar == GARs.BRISTLE) {
                    val tw = network.outputLayer.paramTable()
                    for (index in 0 until tw["W"]!!.columns()) {
                        tw.getValue("W").putColumn(index, cw.getColumn(index.toLong()).dup())
                    }
                    network.outputLayer.setParamTable(tw)
                    evaluationScript.invoke()
                } else {
                    evaluationScript.invoke()
                }
            }
            newOtherModels.clear()

            // RESET TW
            if (gar == GARs.BRISTLE) {
                val tw = network.outputLayer.paramTable()
                for (index in 0 until cw.columns()) {
                    tw.getValue("W").putColumn(index, cw.getColumn(index.toLong()).dup())
                }
                network.outputLayer.setParamTable(tw)
            }

            oldParams = if (gar == GARs.BRISTLE) network.outputLayer.paramTable().getValue("W").dup() else network.params().dup()


            // FIT NETWORK
            try {
                network.fit(iterTrain.next())
            } catch (e: NoSuchElementException) {
                iterTrain.reset()
                epoch++
            }

            // UPDATE CW
            if (gar == GARs.BRISTLE) {
                val tw = network.outputLayer.paramTable().getValue("W")
                for (index in usedClassIndices) {
                    cw.putColumn(index, tw.getColumn(index.toLong()).dup())
                }
            }

            // SHARE MODEL
            if (iteration % iterationsBeforeSending == 0) {
                val message = craftMessage(if (gar == GARs.BRISTLE) cw.dup() else network.params().dup(), behavior, random)
                sendModelToPeers(message, iteration, communicationPattern, countPerPeer)
                while (!udpEndpoint.noPendingTFTPMessages()) {
                    logger.debug { "Waiting for all messages to be sent" }
                    delay(200)
                }
            }

            if (iteration % iterationsBeforeEvaluation == 0) {
                // Test
                logger.debug { "Evaluating network " }
                val elapsedTime = System.currentTimeMillis() - start
                val extraElements = mapOf(
                    Pair("before or after averaging", "before"),
                    Pair("#peers included in current batch", "-")
                )
                sendEvaluationToMaster(
                    evaluationProcessor.evaluate(
                        iterTestFull,
                        network,
                        extraElements,
                        elapsedTime,
                        iteration,
                        epoch,
                        0,
                        true
                    )
                )
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
        sendCompletionToMaster(true)
    }

    private fun sendEvaluationToMaster(evaluation: String) {
        community.sendToMaster(MessageId.MSG_NOTIFY_EVALUATION, MsgNotifyEvaluation(evaluation), reliable = false)
    }

    private fun sendCompletionToMaster(success: Boolean) {
        community.sendToMaster(MessageId.MSG_NOTIFY_FINISHED, MsgNotifyFinished(success), reliable = true)
    }

    private fun sendModelToPeers(
        message: INDArray,
        iteration: Int,
        communicationPattern: CommunicationPatterns,
        countPerPeer: Map<Int, Int>,
    ) {
        val sendMessage = when (communicationPattern) {
            CommunicationPatterns.ALL -> community::sendToAll
            CommunicationPatterns.RANDOM -> community::sendToRandomPeer
            CommunicationPatterns.RANDOM_3 -> community::sendToRandomPeer3
            CommunicationPatterns.RR -> community::sendToNextPeerRR
            CommunicationPatterns.RING -> community::sendToNextPeerRing
        }
        sendMessage(MessageId.MSG_PARAM_UPDATE, MsgParamUpdate(message, iteration), countPerPeer.keys, false)
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        logger.debug { "onMessageReceived: ${peer.address}" }
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                logger.debug { "MSG_PARAM_UPDATE ${peer.address.port} ${newOtherModels.size}" }
                val paramUpdate = payload as MsgParamUpdate
                synchronized(newOtherModels) {
                    newOtherModels[peer.address.port] = paramUpdate
                }
            }
            MessageId.MSG_PSI_CA_CLIENT_TO_SERVER -> scope.launch(Dispatchers.IO) {
                logger.debug { "MSG_PSI_CA_CLIENT_TO_SERVER" }
                deferred.await()
                logger.debug { "MSG_PSI_CA_CLIENT_TO_SERVER after await" }
                logger.debug { "1" }
                val (reEncryptedLabels, filter) = serverRespondsClientRequests(
                    labels,
                    payload as MsgPsiCaClientToServer,
                    sraKeyPair
                )
                logger.debug { "2" }
                val port = community.myEstimatedWan.port
                val message = MsgPsiCaServerToClient(reEncryptedLabels, filter, port)
                logger.debug { "Sending MSG_PSI_CA_SERVER_TO_CLIENT to ${peer.address}" }
                community.sendToPeer(peer, MessageId.MSG_PSI_CA_SERVER_TO_CLIENT, message)
                logger.debug { "3" }
                psiRequests.add(peer.address.port)
                logger.debug { Arrays.toString(psiRequests.toArray()) }
                logger.debug { psiRequests.size }
            }
            MessageId.MSG_PSI_CA_SERVER_TO_CLIENT -> {
                logger.debug { "MSG_PSI_CA_SERVER_TO_CLIENT" }
                psiCaMessagesFromServers.add(payload as MsgPsiCaServerToClient)
            }
            MessageId.MSG_NEW_TEST_COMMAND -> scope.launch(Dispatchers.IO) {
                logger.debug { "MSG_NEW_TEST_COMMAND" }
                val seed = community.myEstimatedWan.port
                random = Random(seed)
                sraKeyPair = SRAKeyPair.create(bigPrime, java.util.Random(seed.toLong()))
                val message = (payload as MsgNewTestCommand)
                val mlConfiguration = message.parsedConfiguration
                val figureName = message.figureName
                val dataset = mlConfiguration.dataset
                val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
                val behavior = mlConfiguration.trainConfiguration.behavior
                val (iterTrain, iterTest, iterTestFull) = getDataSetIterators(
                    dataset.inst,
                    datasetIteratorConfiguration,
                    seed.toLong(),
                    baseDirectory,
                    behavior
                )
                val evaluationProcessor = EvaluationProcessor(
                    baseDirectory,
                    "distributed",
                    listOf(
                        "before or after averaging",
                        "#peers included in current batch"
                    )
                )
                evaluationProcessor.newSimulation(
                    "$figureName - ${mlConfiguration.trainConfiguration.gar.id}",
                    listOf(mlConfiguration),
                    false
                )
                val fromTransfer = mlConfiguration.trainConfiguration.transfer
                val network = if (fromTransfer) {
                    loadFromTransferNetwork(File(baseDirectory, "transfer-${dataset.id}"), mlConfiguration.nnConfiguration, seed, dataset.architecture)
                } else {
                    generateNetwork(dataset.architecture, mlConfiguration.nnConfiguration, seed, NNConfigurationMode.REGULAR)
                }
                network.outputLayer.params().muli(0)
                network.setListeners(ScoreIterationListener(printScoreIterations))

                val port = community.myEstimatedWan.port
                val numPeers = community.getPeers().size
                labels = iterTrain.labels
                val countPerPeer = getSimilarPeers(port, labels, sraKeyPair, numPeers, psiCaMessagesFromServers)

                while (psiRequests.size < numPeers) {
                    logger.debug { "Waiting for other peers to finish PSI_CA: found ${psiRequests.size}" }
                    delay(500)
                }
                logger.debug { "All peers finished PSI_CA!" }

                val distribution = datasetIteratorConfiguration.distribution
                val usedClassIndices = distribution.mapIndexed { ind, v -> if (v > 0) ind else null }.filterNotNull()

                trainTestSendNetwork(
                    network,

                    iterTrain,
                    mlConfiguration.trainConfiguration,
                    mlConfiguration.modelPoisoningConfiguration,

                    evaluationProcessor,
                    iterTestFull,

                    iterTest,
                    countPerPeer,
                    usedClassIndices
                )
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }

    private fun loadFromTransferNetwork(transferFile: File, nnConfiguration: NNConfiguration, seed: Int, generateArchitecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration): MultiLayerNetwork {
        val transferNetwork = ModelSerializer.restoreMultiLayerNetwork(transferFile)
        val frozenNetwork = generateNetwork(generateArchitecture, nnConfiguration, seed, NNConfigurationMode.FROZEN)
        for ((k, v) in transferNetwork.paramTable()) {
            if (k.split("_")[0].toInt() < transferNetwork.layers.size - 1) {
                frozenNetwork.setParam(k, v.dup())
            }
        }
        return frozenNetwork
    }
}
