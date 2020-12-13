package nl.tudelft.trustchain.fedml.ai

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.exception.DL4JInvalidInputException
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.IEvaluation
import org.nd4j.evaluation.classification.Evaluation
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicLong
import kotlin.collections.ArrayList
import kotlin.concurrent.fixedRateTimer

private const val DATE_PATTERN = "yyyy-MM-dd_HH.mm.ss"
private val DATE_FORMAT = SimpleDateFormat(DATE_PATTERN, Locale.US)
private val logger = KotlinLogging.logger("SimulatedRunner")

class EvaluationProcessor(
    baseDirectory: File,
    runner: String,
    private val extraElementNames: List<String>
) : CustomEvaluationCallback {
    @Transient
    private val dataLines = ArrayList<Array<String>>()
    @Transient
    private val configurationLines = ArrayList<String>()
    private val fileDirectory = File(baseDirectory.path, "evaluations")
    private val fileResults = File(fileDirectory, "evaluation-$runner-${DATE_FORMAT.format(Date())}.csv")
    private var fileMeta = File(fileDirectory, "evaluation-$runner-${DATE_FORMAT.format(Date())}.meta.csv")
    private lateinit var currentName: String

    data class EvaluationData(
        val beforeAfterAveraging: String,
        val numPeers: String,
        val elapsedTime: Long,
        val iterationCount: Int,
        val epoch: Int
    )

    init {
        if (!fileDirectory.exists()) {
            fileDirectory.mkdirs()
        }
        fileResults.createNewFile()
        fileMeta.createNewFile()
        configurationLines.add(
            arrayOf(
                "name",
                "simulationIndex",
                "dataset",
                "optimizer",
                "learning rate",
                "momentum",
                "l2",

                "batchSize",
                "iteratorDistribution",
                "maxTestSamples",

                "gar",
                "communicationPattern",
                "behavior",
                "numEpochs",
                "slowdown",
                "joiningLate",

                "local model poisoning attack",
                "#attackers",
            ).joinToString(",")
        )

        val mainDataLineNames = arrayOf(
            "name",
            "simulationIndex",
            "elapsedTime",
            "epoch",
            "iteration",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "gmeasure",
            "mcc",
            "score"
        )
        dataLines.add(Array(mainDataLineNames.size + extraElementNames.size) { "" })
        System.arraycopy(mainDataLineNames, 0, dataLines[0], 0, mainDataLineNames.size)
        for ((index, name) in extraElementNames.withIndex()) {
            dataLines[0][mainDataLineNames.size + index] = name
        }

        fixedRateTimer(period = 2500) {
            PrintWriter(fileResults).use { pw ->
                synchronized(dataLines) {
                    dataLines
                        .map(::convertToCSV)
                        .forEach(pw::println)
                }
            }
        }
    }

    private fun convertToCSV(data: Array<String>): String {
        return data.joinToString(", ") { escapeSpecialCharacters(it) }
    }

    private fun escapeSpecialCharacters(data: String): String {
        return if (data.contains(",") || data.contains("\"") || data.contains("'")) {
            "\"" + data.replace("\"", "\"\"") + "\""
        } else {
            data.replace("\\R".toRegex(), " ")
        }
    }

    override fun newSimulation(
        name: String,
        mlConfiguration: List<MLConfiguration>
    ) {
        this.currentName = name
        mlConfiguration.forEachIndexed { index, configuration ->
            val nnConfiguration = configuration.nnConfiguration
            val datasetIteratorConfiguration = configuration.datasetIteratorConfiguration
            val trainConfiguration = configuration.trainConfiguration
            val modelPoisoningConfiguration = configuration.modelPoisoningConfiguration
            configurationLines.add(
                arrayOf(
                    name,
                    index.toString(),
                    configuration.dataset.text,
                    nnConfiguration.optimizer.text,
                    nnConfiguration.learningRate.text,
                    nnConfiguration.momentum?.text ?: "<null>",
                    nnConfiguration.l2.text,

                    datasetIteratorConfiguration.batchSize.text,
                    datasetIteratorConfiguration.distribution.toString().replace(", ", "-"),
                    datasetIteratorConfiguration.maxTestSamples.text,

                    trainConfiguration.gar.text,
                    trainConfiguration.communicationPattern.text,
                    trainConfiguration.behavior.text,
                    trainConfiguration.numEpochs.text,
                    trainConfiguration.slowdown.text,
                    trainConfiguration.joiningLate.text,

                    modelPoisoningConfiguration.attack.text,
                    modelPoisoningConfiguration.numAttackers.text,
                ).joinToString(",")
            )
        }
        PrintWriter(fileMeta).use { pw ->
            configurationLines.forEach(pw::println)
        }
    }

    override fun call(
        model: Model,
        evaluations: Array<out IEvaluation<*>>,
        simulationIndex: Int,
        score: Double,
        extraElements: Map<String, String>,
        elapsedTime: Long,
        iterations: Int,
        epoch: Int
    ) {
        val accuracy = evaluations[0].getValue(Evaluation.Metric.ACCURACY)
        val f1 = evaluations[0].getValue(Evaluation.Metric.F1)
        val precision = evaluations[0].getValue(Evaluation.Metric.PRECISION)
        val recall = evaluations[0].getValue(Evaluation.Metric.RECALL)
        val gMeasure = evaluations[0].getValue(Evaluation.Metric.GMEASURE)
        val mcc = evaluations[0].getValue(Evaluation.Metric.MCC)
        val mainDataLineElements = arrayOf(
            this.currentName,
            simulationIndex.toString(),
            elapsedTime.toString(),
            epoch.toString(),
            iterations.toString(),
            accuracy.toString(),
            f1.toString(),
            precision.toString(),
            recall.toString(),
            gMeasure.toString(),
            mcc.toString(),
            score.toString()
        )
        synchronized(dataLines) {
            dataLines.add(Array(mainDataLineElements.size + extraElements.size) { "" })
            System.arraycopy(mainDataLineElements, 0, dataLines.last(), 0, mainDataLineElements.size)
            for ((name, value) in extraElements) {
                dataLines.last()[mainDataLineElements.size + extraElementNames.indexOf(name)] = value
            }
        }
    }

    fun done() {
        synchronized(dataLines) {
            dataLines.add(Array(dataLines[0].size) { "DONE" })

            PrintWriter(fileResults).use { pw ->
                dataLines
                    .map(::convertToCSV)
                    .forEach(pw::println)
            }
        }
    }

    fun evaluate(testDataSetIterator: CustomBaseDatasetIterator, network: MultiLayerNetwork, extraElements: Map<String, String>, elapsedTime: Long, iterations: Int, epoch: Int, simulationIndex: Int, logging: Boolean) {
        testDataSetIterator.reset()
        val evaluations = arrayOf(Evaluation())

        if (logging) logger.debug { "Starting evaluation, #iterations = $iterations, ${extraElements.getOrDefault("before or after averaging", "")}" }
        network.doEvaluation(testDataSetIterator, *evaluations)

        for (evaluation in evaluations) {
            if (logging) logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }
        network.computeGradientAndScore()
        call(network, evaluations, simulationIndex, network.score(), extraElements, elapsedTime, iterations, epoch)
    }
}
