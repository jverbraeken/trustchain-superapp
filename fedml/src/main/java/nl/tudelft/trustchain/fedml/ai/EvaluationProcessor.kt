package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.deeplearning4j.optimize.listeners.callbacks.EvaluationCallback
import org.nd4j.evaluation.IEvaluation
import org.nd4j.evaluation.classification.Evaluation
import java.io.File
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*

private const val DATE_PATTERN = "yyyy-MM-dd_HH.mm.ss"
private val DATE_FORMAT = SimpleDateFormat(DATE_PATTERN, Locale.US)

class EvaluationProcessor(
    baseDirectory: File,
    runner: String,
    mlConfiguration: MLConfiguration,
    seed: Int,
    private val extraElementNames: List<String>,
    filenameAddition: String = ""
) : EvaluationCallback {
    private val dataLines = ArrayList<Array<String>>()
    private val fileDirectory = File(baseDirectory.path, "evaluations")
    private val fileResults = File(fileDirectory, "evaluation$filenameAddition-${DATE_FORMAT.format(Date())}.csv")
    private var fileMeta = File(fileDirectory, "evaluation$filenameAddition-${DATE_FORMAT.format(Date())}.meta.csv")
    internal var epoch = 0
    internal var iteration = 0
    internal var extraElements = mapOf<String, String>()
    internal var elapsedTime = 0L

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
        val nnConfiguration = mlConfiguration.nnConfiguration
        val datasetIteratorConfiguration = mlConfiguration.datasetIteratorConfiguration
        val trainConfiguration = mlConfiguration.trainConfiguration
        val modelPoisoningConfiguration = mlConfiguration.modelPoisoningConfiguration
        PrintWriter(fileMeta).use { pw ->
            arrayOf(
                "runner, $runner",
                "dataset, ${mlConfiguration.dataset.text}",
                "optimizer, ${nnConfiguration.optimizer.text}",
                "learning rate, ${nnConfiguration.learningRate.text}",
                "momentum, ${nnConfiguration.momentum?.text ?: "<null>"}",
                "l2, ${nnConfiguration.l2.text}",

                "batchSize, ${datasetIteratorConfiguration.batchSize.text}",
                "iteratorDistribution, ${datasetIteratorConfiguration.distribution}",
                "maxTestSamples, ${datasetIteratorConfiguration.maxTestSamples.text}",

                "gar, ${trainConfiguration.gar.text}",
                "communicationPattern, ${trainConfiguration.communicationPattern.text}",
                "behavior, ${trainConfiguration.behavior.text}",
                "numEpochs, ${trainConfiguration.numEpochs}",

                "local model poisoning attack, ${modelPoisoningConfiguration.attack.text}",
                "#attackers, ${modelPoisoningConfiguration.numAttackers.text}",

                "seed, $seed"
            ).forEach(pw::println)
        }

        val mainDataLineNames = arrayOf(
            "elapsedTime",
            "epoch",
            "iteration",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "gmeasure",
            "mcc"
        )
        dataLines.add(Array(mainDataLineNames.size + extraElementNames.size) { "" })
        System.arraycopy(mainDataLineNames, 0, dataLines[0], 0, mainDataLineNames.size)
        for ((index, name) in extraElementNames.withIndex()) {
            dataLines[0][mainDataLineNames.size + index] = name
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

    override fun call(
        listener: EvaluativeListener,
        model: Model,
        invocationsCount: Long,
        evaluations: Array<out IEvaluation<IEvaluation<*>>>
    ) {
        val accuracy = evaluations[0].getValue(Evaluation.Metric.ACCURACY)
        val f1 = evaluations[0].getValue(Evaluation.Metric.F1)
        val precision = evaluations[0].getValue(Evaluation.Metric.PRECISION)
        val recall = evaluations[0].getValue(Evaluation.Metric.RECALL)
        val gMeasure = evaluations[0].getValue(Evaluation.Metric.GMEASURE)
        val mcc = evaluations[0].getValue(Evaluation.Metric.MCC)
        val mainDataLineElements = arrayOf(
            elapsedTime.toString(),
            epoch.toString(),
            iteration.toString(),
            accuracy.toString(),
            f1.toString(),
            precision.toString(),
            recall.toString(),
            gMeasure.toString(),
            mcc.toString()
        )
        dataLines.add(Array(mainDataLineElements.size + extraElements.size) { "" })
        System.arraycopy(mainDataLineElements, 0, dataLines.last(), 0, mainDataLineElements.size)
        for ((name, value) in extraElements) {
            dataLines.last()[mainDataLineElements.size + extraElementNames.indexOf(name)] = value
        }

        PrintWriter(fileResults).use { pw ->
            dataLines.stream()
                .map(::convertToCSV)
                .forEach(pw::println)
        }
    }

    fun done() {
        dataLines.add(Array(dataLines[0].size) { "DONE" })

        PrintWriter(fileResults).use { pw ->
            dataLines.stream()
                .map(::convertToCSV)
                .forEach(pw::println)
        }
    }
}
