package nl.tudelft.trustchain.fedml.ai

import nl.tudelft.ipv8.Community
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray

class MNISTSimulatedRunner() : MNISTRunner() {
    override val batchSize = 4

    override fun run() {
        val networks = arrayOf(MultiLayerNetwork(nnConf), MultiLayerNetwork(nnConf))
        networks.forEach {
            it.init()
            it.setListeners(ScoreIterationListener(printScoreIterations))
        }
        while (true) {
            for (net in networks) {
                for (i in 0 until batchSize)
                net.fit(mnistTrain.next())
            }
            var arr: INDArray? = null
            for (i in networks.indices) {
                if (i == 0) {
                    arr = networks[i].params()
                } else {
                    arr = arr!!.add(networks[i].params())
                }
            }
            arr = arr!!.divi(networks.size)
            for (net in networks) {
                net.setParameters(arr!!)
            }
        }
    }

}
