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
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.concurrent.thread
import kotlin.random.Random

private val logger = KotlinLogging.logger("DistributedRunner")
private const val SIZE_RECENT_OTHER_MODELS = 20

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    internal lateinit var baseDirectory: File
    private val newOtherModels = ConcurrentHashMap<Int, INDArray>()
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
        mlConfiguration: MLConfiguration
    ) {}

    private suspend fun getSimilarPeers(
        port: Int,
        labels: List<String>,
        sraKeyPair: SRAKeyPair,
        numPeers: Int,
        psiCaMessagesFromServers: CopyOnWriteArrayList<MsgPsiCaServerToClient>
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

        while (psiCaMessagesFromServers.size < numPeers) {
            delay(200)
        }

        return clientReceivesServerResponses(
            port,
            psiCaMessagesFromServers,
            sraKeyPair
        )
    }

    private fun trainTestSendNetwork(
        // General information
        network: MultiLayerNetwork,

        // Evaluation results
        evaluationProcessor: EvaluationProcessor,
        fullTestDataSetIterator: CustomDataSetIterator,

        // Training the network
        trainDataSetIterator: CustomDataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Integrating and distributing information to peers
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
    ) {
        /**
         * Joining late logic...
         */


        val gar = trainConfiguration.gar.obj
        var iterations = 0
        var iterationsToEvaluation = 0
        var iterationsToSending = 0
        var epoch = 0
        val start = System.currentTimeMillis()

        epochLoop@ while (true) {
            logger.debug { "Starting epoch: $epoch" }
            trainDataSetIterator.reset()
            var oldParams = network.params().dup()

            while (true) {
                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                }
                if (iterationsToSending >= iterationsBeforeSending) {
                    iterationsToSending = 0
                }
                var endEpoch = false

                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                val newParams = network.params().dup()
                val gradient = oldParams.sub(newParams)
                iterations++
                iterationsToEvaluation++
                iterationsToSending++

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    // Test
                    logger.debug { "Evaluating network " }
                    val elapsedTime = System.currentTimeMillis() - start
                    val extraElements = mapOf(
                        Pair("before or after averaging", "before"),
                        Pair("#peers included in current batch", "-")
                    )
                    sendEvaluationToMaster(evaluationProcessor.evaluate(
                        fullTestDataSetIterator,
                        network,
                        extraElements,
                        elapsedTime,
                        iterations,
                        epoch,
                        0,
                        true
                    ))

                    if (iterationsToSending >= iterationsBeforeSending) {
                        // Attack
                        val attack = modelPoisoningConfiguration.attack
                        val attackVectors = attack.obj.generateAttack(
                            modelPoisoningConfiguration.numAttackers,
                            oldParams,
                            gradient,
                            newOtherModels,
                            random
                        )
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
                                newOtherModels,
                                recentOtherModels,
                                testDataSetIterator,
                                countPerPeer,
                                true
                            )
                            recentOtherModels.addAll(newOtherModels.toList())
                            while (recentOtherModels.size > SIZE_RECENT_OTHER_MODELS) {
                                recentOtherModels.removeFirst()
                            }
                            newOtherModels.clear()
                            network.setParameters(averageParams)
                            val elapsedTime2 = System.currentTimeMillis() - start
                            val extraElements2 = mapOf(
                                Pair("before or after averaging", "after"),
                                Pair("#peers included in current batch", numPeers.toString())
                            )
                            sendEvaluationToMaster(evaluationProcessor.evaluate(
                                fullTestDataSetIterator,
                                network,
                                extraElements2,
                                elapsedTime2,
                                iterations,
                                epoch,
                                0,
                                true
                            ))
                        }
                        // Send new parameters to other peers
                        sendModelToPeers(
                            averageParams,
                            trainConfiguration.behavior,
                            trainConfiguration.communicationPattern,
                            countPerPeer
                        )
                    }
                }
                oldParams = network.params().dup()
                if (iterations >= trainConfiguration.maxIteration.value) {
                    break@epochLoop
                }
                if (endEpoch) {
                    epoch++
                    break
                }
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
        averageParams: INDArray,
        behavior: Behaviors,
        communicationPattern: CommunicationPatterns,
        countPerPeer: Map<Int, Int>
    ) {
        val message = craftMessage(averageParams, behavior, random)
        val sendMessage = when (communicationPattern) {
            CommunicationPatterns.ALL -> community::sendToAll
            CommunicationPatterns.RANDOM -> community::sendToRandomPeer
            CommunicationPatterns.RR -> community::sendToNextPeerRR
            CommunicationPatterns.RING -> community::sendToNextPeerRing
        }
        sendMessage(MessageId.MSG_PARAM_UPDATE, MsgParamUpdate(message), countPerPeer.keys, false)
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        logger.debug { "onMessageReceived" }
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                logger.debug { "MSG_PARAM_UPDATE" }
                val paramUpdate = payload as MsgParamUpdate
                newOtherModels[peer.address.port] = paramUpdate.array
            }
            MessageId.MSG_PSI_CA_CLIENT_TO_SERVER -> scope.launch(Dispatchers.IO) {
                logger.debug { "MSG_PSI_CA_CLIENT_TO_SERVER" }
                deferred.await()
                val (reEncryptedLabels, filter) = serverRespondsClientRequests(
                    labels,
                    payload as MsgPsiCaClientToServer,
                    sraKeyPair
                )
                val port = community.myEstimatedWan.port
                val message = MsgPsiCaServerToClient(reEncryptedLabels, filter, port)
                community.sendToPeer(peer, MessageId.MSG_PSI_CA_SERVER_TO_CLIENT, message)
            }
            MessageId.MSG_PSI_CA_SERVER_TO_CLIENT -> {
                logger.debug { "MSG_PSI_CA_SERVER_TO_CLIENT" }
                psiCaMessagesFromServers.add(payload as MsgPsiCaServerToClient)
            }
            MessageId.MSG_NEW_TEST_COMMAND -> scope.launch {
                logger.debug { "MSG_NEW_TEST_COMMAND" }
                val seed = community.myEstimatedWan.port
                random = Random(seed)
                sraKeyPair = SRAKeyPair.create(bigPrime, java.util.Random(seed.toLong()))
                val mlConfiguration = (payload as MsgNewTestCommand).parsedConfiguration
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
                evaluationProcessor.newSimulation("distributed simulation", listOf(mlConfiguration))
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

                trainTestSendNetwork(
                    network,

                    evaluationProcessor,
                    iterTestFull,

                    iterTrain,
                    mlConfiguration.trainConfiguration,
                    mlConfiguration.modelPoisoningConfiguration,

                    iterTest,
                    countPerPeer
                )
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }
}
