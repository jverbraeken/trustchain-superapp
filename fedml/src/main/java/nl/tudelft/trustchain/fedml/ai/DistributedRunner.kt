package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MessageListener
import nl.tudelft.trustchain.fedml.ipv8.MsgParamUpdate
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File

private val logger = KotlinLogging.logger("DistributedRunner")

class DistributedRunner(private val community: FedMLCommunity) : Runner(),
    MessageListener {
    private val paramBuffer: MutableList<Pair<INDArray, Int>> = ArrayList()
    private val aggregationRule: AggregationRule = Mozi()

    override fun run(
        baseDirectory: File,
        numEpochs: Epochs,
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes,
        iteratorDistribution: IteratorDistributions,
        maxTestSamples: MaxTestSamples,
        seed: Int
    ) {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                dataset,
                batchSize,
                iteratorDistribution,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                dataset,
                batchSize,
                iteratorDistribution,
                seed,
                maxTestSamples
            )
//            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "distributed",
                dataset.text,
                optimizer.text,
                learningRate.text,
                momentum?.text ?: "null",
                l2.text,
                batchSize.text,
                listOf(
                    "before or after averaging",
                    "total samples",
                    "#peers included in current batch"
                )
            )
            val network = generateNetwork(dataset, optimizer, learningRate, momentum, l2, seed)
            network.setListeners(ScoreIterationListener(printScoreIterations))

            trainNetwork(
                network,
                evaluationProcessor,
                trainDataSetIterator,
                testDataSetIterator,
                numEpochs,
                batchSize
            )
        }
    }

    private fun trainNetwork(
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        trainDataSetIterator: DataSetIterator,
        testDataSetIterator: DataSetIterator,
        numEpochs: Epochs,
        batchSize: BatchSizes
    ) {
        var samplesCounter = 0
        var epoch = 0
        var iterations = 0
        var iterationsToEvaluation = 0
        for (i in 0 until numEpochs.value) {
            epoch++
            trainDataSetIterator.reset()
            logger.debug { "Starting epoch: $epoch" }
            evaluationProcessor.epoch = epoch
            val start = System.currentTimeMillis()
            while (true) {
                var endEpoch = false
                try {
                    network.fit(trainDataSetIterator.next())
                } catch (e: NoSuchElementException) {
                    endEpoch = true
                }
                samplesCounter += batchSize.value
                iterations += batchSize.value
                iterationsToEvaluation += batchSize.value

                if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                    iterationsToEvaluation = 0
                    val end = System.currentTimeMillis()
                    val newSamplesCounter = evaluateNetwork(
                        testDataSetIterator,
                        evaluationProcessor,
                        end - start,
                        iterations,
                        samplesCounter,
                        network,
                        epoch
                    )
                    samplesCounter =
                        if (newSamplesCounter == -1) samplesCounter else newSamplesCounter
                }
                if (endEpoch)
                    break
            }
        }
        logger.debug { "Done training the network" }
        evaluationProcessor.done()
    }

    private fun evaluateNetwork(
        testDataSetIterator: DataSetIterator,
        evaluationProcessor: EvaluationProcessor,
        elapsedTime: Long,
        iterations: Int,
        samplesCounter: Int,
        network: MultiLayerNetwork,
        epoch: Int
    ): Int {
        evaluationProcessor.iteration = iterations
        execEvaluationProcessor(
            evaluationProcessor,
            samplesCounter,
            "",
            elapsedTime,
            testDataSetIterator,
            network,
            network.iterationCount,
            epoch
        )

        var ret = -1
        val start = System.currentTimeMillis()
        val numPeers = paramBuffer.size
        if (numPeers == 0) {
            evaluationProcessor.skip()
        } else {
            val averageParams = aggregationRule.integrateParameters(
                Pair(
                    network.params(),
                    samplesCounter
                ), paramBuffer, network, testDataSetIterator
            )
            ret = averageParams.second
            community.sendToAll(
                MessageId.MSG_PARAM_UPDATE,
                MsgParamUpdate(averageParams.first, samplesCounter),
                true
            )
            paramBuffer.clear()
            network.setParameters(averageParams.first)
            val end = System.currentTimeMillis()

            execEvaluationProcessor(
                evaluationProcessor,
                samplesCounter,
                numPeers.toString(),
                end - start,
                testDataSetIterator,
                network,
                iterations,
                epoch
            )
        }
        return ret
    }

    private fun execEvaluationProcessor(
        evaluationProcessor: EvaluationProcessor,
        samplesCounter: Int,
        numPeers: String,
        elapsedTime: Long,
        testDataSetIterator: DataSetIterator,
        network: MultiLayerNetwork,
        iterationCount: Int,
        epoch: Int
    ) {
        evaluationProcessor.extraElements = mapOf(
            Pair("before or after averaging", "before"),
            Pair("total samples", samplesCounter.toString()),
            Pair("#peers included in current batch", numPeers)
        )
        evaluationProcessor.elapsedTime = elapsedTime
        val evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
        evaluationListener.callback = evaluationProcessor
        evaluationListener.iterationDone(network, iterationCount, epoch)
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
