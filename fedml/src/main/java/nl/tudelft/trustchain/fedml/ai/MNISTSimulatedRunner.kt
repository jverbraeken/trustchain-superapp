package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray

class MNISTSimulatedRunner : MNISTRunner() {
    override val batchSize = 4

    override fun run() {
        val networks = arrayOf(MultiLayerNetwork(nnConf), MultiLayerNetwork(nnConf))
        networks.forEach {
            it.init()
            it.setListeners(ScoreIterationListener(printScoreIterations))
        }
        while (true) {
            for (net in networks) {
                for (i in 0 until batchSize) {
                    net.fit(mnistTrain.next())
                }
            }
            val params : MutableList<Pair<INDArray, Int>> = ArrayList(networks.size)
            networks.forEach { params.add(Pair(it.params(), batchSize)) }
            val averageParams = calculateWeightedAverageParams(params)
            networks.forEach { it.setParams(averageParams)}
        }
    }
}
