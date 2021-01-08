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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import java.io.File
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.collections.ArrayDeque
import kotlin.random.Random

private val logger = KotlinLogging.logger("DistributedRunner")
private const val SIZE_RECENT_OTHER_MODELS = 20
private val psiRequests = ConcurrentHashMap.newKeySet<Int>()

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    internal lateinit var baseDirectory: File
    private val newOtherModels = ConcurrentHashMap<Int, MsgParamUpdate>()
    private val recentOtherModels = ArrayDeque<Pair<Int, INDArray>>()
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
//            if (count > 8000) {
//                throw IllegalArgumentException("Didn't succeed in finding similar peers")
//            }
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
        trainDataSetIterator: CustomDataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Evaluation results
        evaluationProcessor: EvaluationProcessor,
        fullTestDataSetIterator: CustomDataSetIterator,

        // Integrating and distributing information to peers
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
    ) {
        /**
         * Joining late logic...
         */

        val gar = trainConfiguration.gar.obj
        var epochEnd = true
        var epoch = -1
        val start = System.currentTimeMillis()
        var oldParams: INDArray = NDArray()
        val udpEndpoint = community.endpoint.udpEndpoint!!
        val joiningLateRounds = trainConfiguration.joiningLate.rounds
        val slowdown = ((1.0 / trainConfiguration.slowdown.multiplier) - 1).toInt()

        if (joiningLateRounds > 0) {
            while (newOtherModels.none { it.value.iteration >= joiningLateRounds }) {
                logger.debug { "Joining late: ${newOtherModels.map { it.value.iteration }.maxOrNull()};$joiningLateRounds" }
                delay(500)
            }
        }
        val iterationTimeStart = System.currentTimeMillis()
        var iterationTimeEnd: Long? = null

        for (iteration in 0 until trainConfiguration.maxIteration.value) {
            if (epochEnd) {
                epoch++
                logger.debug { "Epoch: $epoch" }
                epochEnd = false
                trainDataSetIterator.reset()
                oldParams = network.params().dup()
            }
            logger.debug { "Iteration: $iteration" }

            try {
                network.fit(trainDataSetIterator.next())
            } catch (e: NoSuchElementException) {
                epochEnd = true
            }
            val newParams = network.params().dup()
            val gradient = oldParams.sub(newParams)

            if (iteration % trainConfiguration.iterationsBeforeEvaluation!! == 0) {
                // Test
                logger.debug { "Evaluating network " }
                val elapsedTime = System.currentTimeMillis() - start
                val extraElements = mapOf(
                    Pair("before or after averaging", "before"),
                    Pair("#peers included in current batch", "-")
                )
                sendEvaluationToMaster(
                    evaluationProcessor.evaluate(
                        fullTestDataSetIterator,
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

            if (iteration % trainConfiguration.iterationsBeforeSending!! == 0) {
                val newOtherModelsWI = newOtherModels.map { Pair(it.key, it.value.array) }.toMap()
                // Attack
                val attack = modelPoisoningConfiguration.attack
                val attackVectors = attack.obj.generateAttack(
                    modelPoisoningConfiguration.numAttackers,
                    oldParams,
                    gradient,
                    newOtherModelsWI,
                    random
                ).map { Pair(it.key, MsgParamUpdate(it.value, -1)) }
                newOtherModels.putAll(attackVectors)

                // Integrate parameters of other peers
                val numPeers = newOtherModels.size + 1
                val averageParams: INDArray
                if (numPeers == 1) {
                    logger.debug { "No received params => skipping integration evaluation" }
                    averageParams = newParams
                    network.setParameters(averageParams)
                } else {
                    logger.debug { "Params received => executing aggregation rule" }
                    averageParams = gar.integrateParameters(
                        network,
                        oldParams,
                        gradient,
                        newOtherModelsWI,
                        recentOtherModels,
                        testDataSetIterator,
                        countPerPeer,
                        true
                    )
                    recentOtherModels.addAll(newOtherModelsWI.toList())
                    while (recentOtherModels.size > SIZE_RECENT_OTHER_MODELS) {
                        recentOtherModels.removeFirst()
                    }
                    newOtherModels.clear()
                    network.setParameters(averageParams)
                }
                // Send new parameters to other peers
                val message = craftMessage(averageParams.dup(), trainConfiguration.behavior, random)

                while (!udpEndpoint.noPendingTFTPMessages()) {
                    logger.debug { "Waiting for all TFTP messages to be sent" }
                    delay(500)
                }

                sendModelToPeers(message, iteration, trainConfiguration.communicationPattern, countPerPeer)

                if (iteration % trainConfiguration.iterationsBeforeEvaluation == 0) {
                    val elapsedTime2 = System.currentTimeMillis() - start
                    val extraElements2 = mapOf(
                        Pair("before or after averaging", "after"),
                        Pair("#peers included in current batch", numPeers.toString())
                    )
                    sendEvaluationToMaster(
                        evaluationProcessor.evaluate(
                            fullTestDataSetIterator,
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
            }
            if (iterationTimeEnd == null) {
                iterationTimeEnd = System.currentTimeMillis()
            }
            delay((iterationTimeEnd - iterationTimeStart) * slowdown)
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
                val (iterTrain, iterTrainFull, iterTest, iterTestFull) = getDataSetIterators(
                    dataset,
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
                    listOf(mlConfiguration)
                )
                val network = generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.nnConfiguration,
                    seed
                )
                network.setListeners(ScoreIterationListener(printScoreIterations))

                val port = community.myEstimatedWan.port
                val numPeers = community.getPeers().size
                labels = iterTrain.labels
                val countPerPeer = getSimilarPeers(port, labels, sraKeyPair, numPeers, psiCaMessagesFromServers)

                while (psiRequests.size < 3 - 1) {
                    logger.debug { "Waiting for other peers to finish PSI_CA: found ${psiRequests.size}" }
                    delay(500)
                }
                logger.debug { "All peers finished PSI_CA!" }

                trainTestSendNetwork(
                    network,

                    iterTrain,
                    mlConfiguration.trainConfiguration,
                    mlConfiguration.modelPoisoningConfiguration,

                    evaluationProcessor,
                    iterTestFull,

                    iterTest,
                    countPerPeer
                )
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }
}
