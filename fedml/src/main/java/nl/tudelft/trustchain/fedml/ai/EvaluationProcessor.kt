package nl.tudelft.trustchain.fedml.ai

import kotlinx.coroutines.withTimeoutOrNull
import mu.KotlinLogging
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.evaluation.IEvaluation
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*
import kotlin.concurrent.fixedRateTimer

private const val DATE_PATTERN = "yyyy-MM-dd_HH.mm.ss"
private val DATE_FORMAT = SimpleDateFormat(DATE_PATTERN, Locale.US)
private val logger = KotlinLogging.logger("EvaluationProcessor")

class EvaluationProcessor(
    baseDirectory: File,
    runner: String,
    private val extraElementNames: List<String>
) {
    private val configurationHeader = arrayOf(
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
        "iterationsBeforeEvaluation",
        "iterationsBeforeSending",

        "local model poisoning attack",
        "#attackers"
    ).joinToString(",")

    @Transient
    private val configurationLines = arrayListOf(configurationHeader)

    private val evaluationHeader = arrayOf(
        "environment",
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
        "score",
        "before or after averaging",
        "#peers included in current batch"
    )

    @Transient
    private val evaluationLines = arrayListOf(evaluationHeader)
    private val fileDirectory = File(baseDirectory.path, "evaluations")
    private val fileResults = File(fileDirectory, "evaluation-$runner-${DATE_FORMAT.format(Date())}.csv")
    private var fileMeta = File(fileDirectory, "evaluation-$runner-${DATE_FORMAT.format(Date())}.meta.csv")
    private lateinit var currentName: String

    init {
        if (!fileDirectory.exists()) {
            fileDirectory.mkdirs()
        }
        fileResults.createNewFile()
        fileMeta.createNewFile()

        val newEvaluationHeader = Array(evaluationHeader.size + extraElementNames.size) { "" }
        evaluationHeader.copyInto(newEvaluationHeader)
        for ((index, name) in extraElementNames.withIndex()) {
            newEvaluationHeader[evaluationHeader.size + index] = name
        }
        evaluationLines[0] = newEvaluationHeader

        fixedRateTimer(period = 2500) {
            PrintWriter(fileResults).use { pw ->
                synchronized(evaluationLines) {
                    evaluationLines
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

    fun newSimulation(
        name: String,
        mlConfiguration: List<MLConfiguration>
    ) {
        this.currentName = name
        mlConfiguration.forEachIndexed { index, configuration ->
            val line = parseConfiguration(name, index, configuration)
            configurationLines.add(line)
        }
        PrintWriter(fileMeta).use { pw -> configurationLines.forEach(pw::println) }
    }

    private fun parseConfiguration(name: String, index: Int, configuration: MLConfiguration): String {
        val nnConfiguration = configuration.nnConfiguration
        val datasetIteratorConfiguration = configuration.datasetIteratorConfiguration
        val trainConfiguration = configuration.trainConfiguration
        val modelPoisoningConfiguration = configuration.modelPoisoningConfiguration
        return arrayOf(
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
            trainConfiguration.maxIteration.text,
            trainConfiguration.slowdown.text,
            trainConfiguration.joiningLate.text,
            trainConfiguration.iterationsBeforeEvaluation ?: "<null>",
            trainConfiguration.iterationsBeforeSending ?: "<null>",

            modelPoisoningConfiguration.attack.text,
            modelPoisoningConfiguration.numAttackers.text,
        ).joinToString(",")
    }

    fun call(
        model: Model,
        evaluations: Array<out IEvaluation<*>>,
        simulationIndex: Int,
        score: Double,
        extraElements: Map<String, String>,
        elapsedTime: Long,
        iterations: Int,
        epoch: Int
    ): String {
        val accuracy = evaluations[0].getValue(Evaluation.Metric.ACCURACY)
        val f1 = evaluations[0].getValue(Evaluation.Metric.F1)
        val precision = evaluations[0].getValue(Evaluation.Metric.PRECISION)
        val recall = evaluations[0].getValue(Evaluation.Metric.RECALL)
        val gMeasure = evaluations[0].getValue(Evaluation.Metric.GMEASURE)
        val mcc = evaluations[0].getValue(Evaluation.Metric.MCC)
        val dataLineElements = arrayOf(
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
        synchronized(evaluationLines) {
            evaluationLines.add(Array(dataLineElements.size + extraElements.size) { "" })
            System.arraycopy(dataLineElements, 0, evaluationLines.last(), 0, dataLineElements.size)
            for ((name, value) in extraElements) {
                evaluationLines.last()[dataLineElements.size + extraElementNames.indexOf(name)] = value
            }
        }
        return convertToCSV(evaluationLines.last())
    }

    fun done() {
        synchronized(evaluationLines) {
            evaluationLines.add(Array(evaluationLines[0].size) { "DONE" })

            PrintWriter(fileResults).use { pw ->
                evaluationLines
                    .map(::convertToCSV)
                    .forEach(pw::println)
            }
        }
    }

    fun evaluate(testDataSetIterator: DataSetIterator, network: MultiLayerNetwork, extraElements: Map<String, String>, elapsedTime: Long, iterations: Int, epoch: Int, simulationIndex: Int, logging: Boolean): String {
        testDataSetIterator.reset()
        val evaluations = arrayOf(Evaluation())

        if (logging) logger.debug { "Starting evaluation, #iterations = $iterations, ${extraElements.getOrDefault("before or after averaging", "")}" }
        network.doEvaluation(testDataSetIterator, *evaluations)

        for (evaluation in evaluations) {
            if (logging) logger.debug { "${evaluation.javaClass.simpleName}:\n${evaluation.stats()}" }
        }
        return call(network, evaluations, simulationIndex, network.score(), extraElements, elapsedTime, iterations, epoch)
    }
}
