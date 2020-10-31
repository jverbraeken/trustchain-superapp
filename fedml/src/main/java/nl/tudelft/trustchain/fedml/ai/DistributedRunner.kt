package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.ai.gar.AggregationRule
import nl.tudelft.trustchain.fedml.ai.gar.Mozi
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

class DistributedRunner(private val community: FedMLCommunity) : Runner(), MessageListener {
    private val paramBuffer: MutableList<Pair<INDArray, Int>> = ArrayList()

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.batchSize,
                mlConfiguration.iteratorDistribution,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.batchSize,
                mlConfiguration.iteratorDistribution,
                seed,
                mlConfiguration.maxTestSamples
            )
//            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
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
                mlConfiguration.optimizer,
                mlConfiguration.learningRate,
                mlConfiguration.momentum,
                mlConfiguration.l2,
                seed
            )
            network.setListeners(ScoreIterationListener(printScoreIterations))

            trainNetwork(
                network,
                evaluationProcessor,
                trainDataSetIterator,
                testDataSetIterator,
                mlConfiguration.epoch,
                mlConfiguration.batchSize,
                mlConfiguration.gar
            )
        }
    }

    private fun trainNetwork(
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        trainDataSetIterator: DataSetIterator,
        testDataSetIterator: DataSetIterator,
        numEpochs: Epochs,
        batchSize: BatchSizes,
        gar: GARs
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
                        network,
                        evaluationProcessor,
                        testDataSetIterator,
                        end - start,
                        iterations,
                        samplesCounter,
                        epoch,
                        gar.obj
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
        network: MultiLayerNetwork,
        evaluationProcessor: EvaluationProcessor,
        testDataSetIterator: DataSetIterator,
        elapsedTime: Long,
        iterations: Int,
        samplesCounter: Int,
        epoch: Int,
        gar: AggregationRule
    ): Int {
        logger.debug { "Evaluating network " }
        evaluationProcessor.iteration = iterations
        execEvaluationProcessor(
            evaluationProcessor,
            testDataSetIterator,
            network,
            EvaluationProcessor.EvaluationData(
                "before",
                samplesCounter,
                "",
                elapsedTime,
                network.iterationCount,
                epoch
            )
        )

        var ret = -1
        val start = System.currentTimeMillis()
        val numPeers = paramBuffer.size + 1
        val averageParams: Pair<INDArray, Int>
        if (numPeers == 1) {
            logger.debug { "No peers => skipping integration evaluation" }
            evaluationProcessor.skip()
            averageParams = Pair(network.params().dup(), samplesCounter)
        } else {
            logger.debug { "Peers found => executing aggregation rule" }
            averageParams = gar.integrateParameters(
                Pair(
                    network.params().dup(),
                    samplesCounter
                ), paramBuffer, network, testDataSetIterator
            )
            ret = averageParams.second
            paramBuffer.clear()
            network.setParameters(averageParams.first)
            val end = System.currentTimeMillis()

            execEvaluationProcessor(
                evaluationProcessor,
                testDataSetIterator,
                network,
                EvaluationProcessor.EvaluationData(
                    "after",
                    samplesCounter,
                    numPeers.toString(),
                    end - start,
                    iterations,
                    epoch
                )
            )
        }
        community.sendToAll(
            MessageId.MSG_PARAM_UPDATE,
            MsgParamUpdate(averageParams.first, samplesCounter),
            true
        )
        return ret
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
