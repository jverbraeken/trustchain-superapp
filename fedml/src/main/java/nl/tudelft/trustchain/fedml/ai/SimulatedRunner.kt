package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.launch
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.gar.AggregationRule
import nl.tudelft.trustchain.fedml.ai.gar.SimpleAggregator
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File

private val logger = KotlinLogging.logger("SimulatedRunner")

class SimulatedRunner : Runner() {
    private val aggregationRule: AggregationRule = SimpleAggregator()

    override fun run(
        baseDirectory: File,
        seed: Int,
        mlConfiguration: MLConfiguration
    ) {
        scope.launch {
            val trainDataSetIterators = arrayOf(
                getTrainDatasetIterator(
                    baseDirectory,
                    mlConfiguration.dataset,
                    mlConfiguration.datasetIteratorConfiguration,
                    Behaviors.BENIGN,
                    seed
                ),
                getTrainDatasetIterator(
                    baseDirectory,
                    mlConfiguration.dataset,
                    mlConfiguration.datasetIteratorConfiguration,
                    Behaviors.BENIGN,
                    seed + 1
                )
            )
            val testDataSetIterator = getTestDatasetIterator(
                baseDirectory,
                mlConfiguration.dataset,
                mlConfiguration.datasetIteratorConfiguration,
                Behaviors.BENIGN,
                seed
            )
            val networks = arrayOf(
                generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.nnConfiguration,
                    seed
                ),
                generateNetwork(
                    mlConfiguration.dataset,
                    mlConfiguration.nnConfiguration,
                    seed + 1
                )
            )
            networks.forEach { it.setListeners(ScoreIterationListener(printScoreIterations)) }

            var evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
            val evaluationProcessor = EvaluationProcessor(
                baseDirectory,
                "simulated",
                mlConfiguration,
                seed,
                listOf("before or after averaging")
            )
            evaluationListener.callback = evaluationProcessor

            var epoch = 0
            var iterations = 0
            var iterationsToEvaluation = 0
            val trainConfiguration = mlConfiguration.trainConfiguration
            val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
            for (i in 0 until trainConfiguration.numEpochs.value) {
                epoch++
                trainDataSetIterators.forEach { it.reset() }
                logger.debug { "Starting epoch: $epoch" }
                evaluationProcessor.epoch = epoch
                val start = System.currentTimeMillis()
                while (true) {
                    var endEpoch = false
                    val oldParams = arrayListOf<INDArray>()
                    val gradient = arrayListOf<INDArray>()
                    for ((j, net) in networks.withIndex()) {
                        oldParams.add(net.params().dup())
                        try {
                            net.fit(trainDataSetIterators[j].next())
                        } catch (e: NoSuchElementException) {
                            endEpoch = true
                        }
                        gradient.add(oldParams.last().sub(net.params().dup()))
                    }
                    iterations += datasetIteratorConfiguration.batchSize.value
                    iterationsToEvaluation += datasetIteratorConfiguration.batchSize.value

                    if (iterationsToEvaluation >= iterationsBeforeEvaluation) {
                        iterationsToEvaluation = 0
                        val end = System.currentTimeMillis()
                        evaluationProcessor.iteration = iterations
                        evaluationProcessor.extraElements =
                            mapOf(Pair("before or after averaging", "before"))
                        evaluationProcessor.elapsedTime = end - start
                        evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                        evaluationListener.callback = evaluationProcessor
                        evaluationListener.iterationDone(networks[0], networks[0].iterationCount, epoch)

                        val params: MutableList<Pair<INDArray, Int>> = ArrayList(networks.size)
                        networks.forEach { params.add(Pair(it.params().dup(), datasetIteratorConfiguration.batchSize.value)) }
                        val averageParams = aggregationRule.integrateParameters(
                            params[0],
                            gradient[0],
                            params.subList(1, params.size),
                            networks[0],
                            testDataSetIterator
                        )
                        networks.forEach { it.setParams(averageParams.first) }

                        evaluationProcessor.extraElements =
                            mapOf(Pair("before or after averaging", "after"))
                        evaluationListener = EvaluativeListener(testDataSetIterator, 999999)
                        evaluationListener.callback = evaluationProcessor
                        evaluationListener.iterationDone(networks[0], iterations, epoch)
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
