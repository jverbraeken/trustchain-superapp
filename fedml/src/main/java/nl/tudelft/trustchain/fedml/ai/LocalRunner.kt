package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import org.deeplearning4j.nn.updater.UpdaterCreator
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.AdamUpdater
import org.nd4j.linalg.learning.config.Adam
import java.io.File

private val logger = KotlinLogging.logger("LocalRunner")

class LocalRunner : Runner() {
    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        scope.launch {
            val trainDataSetIterator = getTrainDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.datasetIteratorConfiguration,
                Behaviors.BENIGN,
                seed
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.datasetIteratorConfiguration,
                Behaviors.BENIGN,
                seed
            )
            val network = generateNetwork(
                mlConfiguration.dataset,
                mlConfiguration.nnConfiguration,
                seed
            )
            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "local",
                mlConfiguration,
                seed,
                ArrayList()
            )
            evaluationListener.callback = evaluationProcessor
            network.setListeners(
                ScoreIterationListener(printScoreIterations),
                evaluationListener
            )

            var epoch = 0
            var iterations = 0
            var iterationsToEvaluation = 0
            val trainConfiguration = mlConfiguration.trainConfiguration
            val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
            for (i in 0 until trainConfiguration.numEpochs.value) {
                epoch++
                trainDataSetIterator.reset()
                logger.debug { "Starting epoch: $epoch" }
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                while (true) {
                    var endEpoch = false
                    try {
//                        network.fit(trainDataSetIterator.next())
                        val batch = trainDataSetIterator.next()
                        val pair = network.calculateGradients(batch.features, batch.labels, null, null)
                        val adam = Adam(mlConfiguration.nnConfiguration.learningRate.schedule)
                        val updater = UpdaterCreator.getUpdater(network)
                        updater.update(network, pair.first, iterations, epoch, network.batchSize(), LayerWorkspaceMgr.noWorkspaces())
                        val params = network.params()
                        params.subi(pair.first.gradient())




                        /*val batch = trainDataSetIterator.next()
                        val pair = network.calculateGradients(batch.features, batch.labels, null, null)
//                        val adam = Adam(mlConfiguration.nnConfiguration.learningRate.schedule)
                        val updater = UpdaterCreator.getUpdater(network)
                        updater.update(network, pair.first, iterations, epoch, network.batchSize(), LayerWorkspaceMgr.noWorkspaces())
                        val params = network.params()
                        params.subi(pair.first.gradient())*/
                    } catch (e: NoSuchElementException) {
                        endEpoch = true
                    }
                    iterations += datasetIteratorConfiguration.batchSize.value
                    iterationsToEvaluation += datasetIteratorConfiguration.batchSize.value

                    if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                        iterationsToEvaluation = 0
                        val end = System.currentTimeMillis()
                        evaluationProcessor.iteration = iterations
                        evaluationProcessor.elapsedTime = end - start
                        evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                        evaluationListener.callback = evaluationProcessor
                        evaluationListener.iterationDone(network, iterations, epoch)
                    }
                    if (endEpoch) {
                        break
                    }
                }
            }
            evaluationProcessor.done()
        }
    }
}
