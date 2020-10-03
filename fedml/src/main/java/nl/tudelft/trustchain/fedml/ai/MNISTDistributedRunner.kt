package nl.tudelft.trustchain.fedml.ai

import android.util.Log
import nl.tudelft.ipv8.Peer
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MessageListener
import nl.tudelft.trustchain.fedml.ipv8.MsgParamUpdate
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray

class MNISTDistributedRunner(private val community: FedMLCommunity) : MNISTRunner(),
    MessageListener {
    override val batchSize = 10
    private val paramBuffer: MutableList<Pair<INDArray, Int>> = ArrayList()

    override fun run() {
        FedMLCommunity.registerMessageListener(MessageId.MSG_PARAM_UPDATE, this)
        val network = MultiLayerNetwork(nnConf)
        network.init()
        network.setListeners(ScoreIterationListener(printScoreIterations)/*, EvaluativeListener(mnistTest, 20)*/)

        var samplesCounter = 0
        while (true) {
            val start = System.currentTimeMillis()
            for (i in 0 until batchSize) {
                network.fit(mnistTrain.next())
                samplesCounter++
            }
            val end = System.currentTimeMillis()
            Log.i("MNISTDistributedRunner", "time to fit network for batch: " + (end - start))
            paramBuffer.add(Pair(network.params(), samplesCounter))
            val averageParams = calculateWeightedAverageParams(paramBuffer)
            samplesCounter = averageParams.second
            community.sendToAll(
                MessageId.MSG_PARAM_UPDATE,
                MsgParamUpdate(averageParams.first, samplesCounter),
                true
            )
            paramBuffer.clear()
            network.setParameters(averageParams.first)
        }
    }

    override fun onMessageReceived(messageId: MessageId, peer: Peer, payload: Any) {
        Log.i("MNISTDistributedRunner", "onMessageReceived")
        when (messageId) {
            MessageId.MSG_PARAM_UPDATE -> {
                val paramUpdate = payload as MsgParamUpdate
                paramBuffer.add(Pair(paramUpdate.array, paramUpdate.weight))
            }
            else -> throw Exception("Other messages should not be listened to...")
        }
    }

}
