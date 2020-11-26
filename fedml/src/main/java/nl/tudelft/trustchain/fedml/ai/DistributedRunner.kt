package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ipv8.*
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.random.Random

private val logger = KotlinLogging.logger("DistributedRunner")
private const val SIZE_RECENT_OTHER_MODELS = 20

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
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
    }

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        this.random = Random(seed)
        sraKeyPair = SRAKeyPair.create(bigPrime, java.util.Random(seed.toLong()))
        scope.launch {
            val dataset = mlConfiguration.dataset
            val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
            val behavior = mlConfiguration.trainConfiguration.behavior
            val trainDataSetIterator = dataset.inst(
                datasetIteratorConfiguration,
                seed.toLong(),
                CustomDataSetType.TRAIN,
                baseDirectory,
                behavior
            )
            val testDataSetIterator = dataset.inst(
                datasetIteratorConfiguration,
                seed.toLong(),
                CustomDataSetType.TEST,
                baseDirectory,
                behavior
            )
            val fullTestDataSetIterator = dataset.inst(
                datasetIteratorConfiguration,
                seed.toLong(),
                CustomDataSetType.FULL_TEST,
                baseDirectory,
                behavior
            )
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "distributed",
                listOf(mlConfiguration),
                listOf(
                    "before or after averaging",
                    "#peers included in current batch"
                )
            )
            val network = generateNetwork(
                mlConfiguration.dataset,
                mlConfiguration.nnConfiguration,
                seed
            )
            network.setListeners(ScoreIterationListener(printScoreIterations))

            val port = community.myEstimatedWan.port
            val numPeers = community.getPeers().size
            labels = trainDataSetIterator.labels
            val countPerPeer = getSimilarPeers(port, labels, sraKeyPair, numPeers, psiCaMessagesFromServers)

            trainTestSendNetwork(
                network,

                evaluationProcessor,
                fullTestDataSetIterator,

                trainDataSetIterator,
                mlConfiguration.trainConfiguration,
                mlConfiguration.modelPoisoningConfiguration,

                testDataSetIterator,
                countPerPeer
            )
        }
    }

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
        fullTestDataSetIterator: CustomBaseDatasetIterator,

        // Training the network
        trainDataSetIterator: DataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration,

        // Integrating and distributing information to peers
        testDataSetIterator: CustomBaseDatasetIterator,
        countPerPeer: Map<Int, Int>,
    ) {
        val batchSize = trainDataSetIterator.batch()
        val gar = trainConfiguration.gar.obj
        var iterations = 0
        var iterationsToEvaluation = 0
        for (epoch in 0 until trainConfiguration.numEpochs.value) {
            logger.debug { "Starting epoch: $epoch" }
            evaluationProcessor.epoch = epoch
            trainDataSetIterator.reset()
            val start = System.currentTimeMillis()
            var oldParams = network.params().dup()
            while (true) {

                // Train
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                val newParams = network.params().dup()
                val gradient = oldParams.sub(newParams)
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
                        fullTestDataSetIterator,
                        network,
                        EvaluationProcessor.EvaluationData(
                            "before", "", end - start, network.iterationCount, epoch
                        )
                    )

                    // Integrate parameters of other peers
                    val attack = modelPoisoningConfiguration.attack
                    val attackVectors = attack.obj.generateAttack(
                        modelPoisoningConfiguration.numAttackers,
                        oldParams,
                        gradient,
                        newOtherModels,
                        random
                    )
                    newOtherModels.putAll(attackVectors)
                    val numPeers = newOtherModels.size + 1
                    val averageParams: INDArray
                    if (numPeers == 1) {
                        logger.debug { "No received params => skipping integration evaluation" }
                        averageParams = newParams
                        network.setParameters(averageParams)
                    } else {
                        logger.debug { "Params received => executing aggregation rule" }

                        val start2 = System.currentTimeMillis()
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
                        val end2 = System.currentTimeMillis()

                        execEvaluationProcessor(
                            evaluationProcessor,
                            fullTestDataSetIterator,
                            network,
                            EvaluationProcessor.EvaluationData(
                                "after", numPeers.toString(), end2 - start2, iterations, epoch
                            )
                        )
                    }

                    // Send new parameters to other peers
                    sendModelToPeers(
                        averageParams,
                        trainConfiguration.behavior,
                        trainConfiguration.communicationPattern,
                        countPerPeer
                    )
                }
                oldParams = network.params().dup()
                if (endEpoch) {
                    break
                }
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
    }

    private fun sendModelToPeers(
        averageParams: INDArray,
        behavior: Behaviors,
        communicationPattern: CommunicationPatterns,
        countPerPeer: Map<Int, Int>
    ) {
        val message = craftMessage(averageParams, behavior)
        val sendMessage = when (communicationPattern) {
            CommunicationPatterns.ALL -> community::sendToAll
            CommunicationPatterns.RANDOM -> community::sendToRandomPeer
            CommunicationPatterns.RR -> community::sendToNextPeerRR
            CommunicationPatterns.RING -> community::sendToNextPeerRing
        }
        sendMessage(MessageId.MSG_PARAM_UPDATE, MsgParamUpdate(message), countPerPeer.keys, false)
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
            Pair("#peers included in current batch", evaluationData.numPeers)
        )
        evaluationProcessor.elapsedTime = evaluationData.elapsedTime
        val evaluationListener = CustomEvaluativeListener(testDataSetIterator, 999999)
        evaluationListener.callback = evaluationProcessor
        evaluationListener.invokeListener(
            network,
            1,
            true
        )
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        logger.debug { "onMessageReceived" }
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                val paramUpdate = payload as MsgParamUpdate
                newOtherModels[peer.address.port] = paramUpdate.array
            }
            MessageId.MSG_PSI_CA_CLIENT_TO_SERVER -> scope.launch(Dispatchers.IO) {
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
                psiCaMessagesFromServers.add(payload as MsgPsiCaServerToClient)
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }
}
