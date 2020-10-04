package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

class LocalRunner : Runner() {
    override fun run(
        dataset: Datasets,
        updater: Updaters,
        learningRate: LearningRates,
        momentum: Momentums?,
        l2: L2Regularizations,
        batchSize: BatchSizes
    ) {
        val network = generateNetwork(dataset, updater, learningRate, momentum, l2)
        network.setListeners(
            ScoreIterationListener(printScoreIterations),
            EvaluativeListener(getTestDatasetIterator(dataset, batchSize), 500)
        )
        network.fit(getTrainDatasetIterator(dataset, batchSize))
    }
}
