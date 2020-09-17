package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

class MNISTLocalRunner : MNISTRunner() {
    override fun run() {
        val network = MultiLayerNetwork(nnConf)
        network.init()
        network.setListeners(ScoreIterationListener(printScoreIterations))
        network.fit(mnistTrain)
    }
}
