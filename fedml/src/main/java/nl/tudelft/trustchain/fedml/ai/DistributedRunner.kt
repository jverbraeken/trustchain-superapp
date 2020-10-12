package nl.tudelft.trustchain.fedml.ai

import android.util.Log
import kotlinx.coroutines.launch
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MessageListener
import nl.tudelft.trustchain.fedml.ipv8.MsgParamUpdate
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File

class DistributedRunner(private val community: FedMLCommunity) : Runner(),
    MessageListener {
    private val paramBuffer: MutableList<Pair<INDArray, Int>> = ArrayList()

    override fun run(
        baseDirectory: File,
        numEpochs: Epochs,
        dataset: Datasets,
        optimizer: Optimizers,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    ) {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(baseDirectory, dataset, batchSize)
            val testDataSetIterator = getTestDatasetIterator(baseDirectory, dataset, batchSize)
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
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
            evaluationListener.callback = evaluationProcessor
            val network = generateNetwork(dataset, optimizer, learningRate, momentum, l2)
            network.setListeners(ScoreIterationListener(printScoreIterations))

            var samplesCounter = 0
            var epoch = 0
            var iterations = 0
            for (i in 0 until numEpochs.value) {
                epoch++
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                while (true) {
                    var endEpoch = false
                    for (j in 0 until batchSize.value) {
                        try {
                            network.fit(trainDataSetIterator.next())
                        } catch (e: NoSuchElementException) {
                            endEpoch = true
                        }
                        samplesCounter++
                    }
                    iterations += batchSize.value
                    var end = System.currentTimeMillis()

                    evaluationProcessor.iteration = iterations
                    evaluationProcessor.extraElements = mapOf(
                        Pair("before or after averaging", "before"),
                        Pair("total samples", samplesCounter.toString()),
                        Pair("#peers included in current batch", "")
                    )
                    evaluationProcessor.elapsedTime = end - start
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(network, network.iterationCount, epoch)

                    paramBuffer.add(Pair(network.params(), samplesCounter))
                    val numPeers = paramBuffer.size
                    val averageParams = calculateWeightedAverageParams(paramBuffer)
                    samplesCounter = averageParams.second
                    community.sendToAll(
                        MessageId.MSG_PARAM_UPDATE,
                        MsgParamUpdate(averageParams.first, samplesCounter),
                        true
                    )
                    paramBuffer.clear()
                    network.setParameters(averageParams.first)
                    end = System.currentTimeMillis()

                    evaluationProcessor.extraElements = mapOf(
                        Pair("before or after averaging", "after"),
                        Pair("total samples", samplesCounter.toString()),
                        Pair("#peers included in current batch", numPeers.toString())
                    )
                    evaluationProcessor.elapsedTime = end - start
                    evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                    evaluationListener.callback = evaluationProcessor
                    evaluationListener.iterationDone(network, iterations, epoch)
                    if (endEpoch)
                        break
                }
            }
            evaluationProcessor.done()
        }
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        Log.i("DistributedRunner", "onMessageReceived")
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                val paramUpdate = payload as MsgParamUpdate
                paramBuffer.add(Pair(paramUpdate.array, paramUpdate.weight))
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }

}
