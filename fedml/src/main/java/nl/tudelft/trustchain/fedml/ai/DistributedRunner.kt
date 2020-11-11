package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.CommunicationPatterns
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MessageListener
import nl.tudelft.trustchain.fedml.ipv8.MsgParamUpdate
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.random.Random

private val logger = KotlinLogging.logger("DistributedRunner")

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    private val newOtherModels: CopyOnWriteArrayList<INDArray> = CopyOnWriteArrayList()
    private val recentOtherModels: ConcurrentLinkedDeque<INDArray> = ConcurrentLinkedDeque()
    private lateinit var random: Random

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
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
                mlConfiguration.modelPoisoningConfiguration
            )
        }
    }

    private fun trainTestSendNetwork(
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        trainDataSetIterator: DataSetIterator,
        testDataSetIterator: DataSetIterator,
        trainConfiguration: TrainConfiguration,
        modelPoisoningConfiguration: ModelPoisoningConfiguration
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
                        testDataSetIterator,
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
                    newOtherModels.addAll(attackVectors)
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
                            oldParams,
                            gradient,
                            newOtherModels,
                            network,
                            testDataSetIterator,
                            recentOtherModels
                        )
                        recentOtherModels.addAll(newOtherModels)
                        while (recentOtherModels.size > 20) {
                            recentOtherModels.removeFirst()
                        }
                        newOtherModels.clear()
                        network.setParameters(averageParams)
                        val end2 = System.currentTimeMillis()

                        execEvaluationProcessor(
                            evaluationProcessor,
                            testDataSetIterator,
                            network,
                            EvaluationProcessor.EvaluationData(
                                "after", numPeers.toString(), end2 - start2, iterations, epoch
                            )
                        )
                    }

                    // Send new parameters to other peers
                    val message = craftMessage(averageParams, trainConfiguration.behavior)
                    val sendMessage = when (trainConfiguration.communicationPattern) {
                        CommunicationPatterns.ALL -> community::sendToAll
                        CommunicationPatterns.RANDOM -> community::sendToRandomPeer
                        CommunicationPatterns.RR -> community::sendToNextPeerRR
                        CommunicationPatterns.RING -> community::sendToNextPeerRing
                    }
                    sendMessage(MessageId.MSG_PARAM_UPDATE, MsgParamUpdate(message))
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
                newOtherModels.add(paramUpdate.array)
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }
}
