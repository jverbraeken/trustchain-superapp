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

class EvaluationProcessor(
    baseDirectory: File,
    runner: String,
    mlConfiguration: MLConfiguration,
    seed: Int,
    private val extraElementNames: List<String>
) : EvaluationCallback {
    private val datePattern = "yyyy-MM-dd_HH.mm.ss"
    private val dataLines: MutableList<Array<String>> = ArrayList()
    private var fileResults: File
    private var fileMeta: File
    internal var epoch: Int = 0
    internal var iteration: Int = 0
    internal var extraElements: Map<String, String> = HashMap()
    internal var elapsedTime: Long = 0

    data class EvaluationData(
        val beforeAfterAveraging: String,
        val samplesCounter: Int,
        val numPeers: String,
        val elapsedTime: Long,
        val iterationCount: Int,
        val epoch: Int
    )

    init {
        fileResults = File(
            baseDirectory.path, "evaluation-" + SimpleDateFormat(
                datePattern,
                Locale.US
            ).format(Date()) + ".csv"
        )
        fileResults.createNewFile()

        fileMeta = File(
            baseDirectory.path, "evaluation-" + SimpleDateFormat(
                datePattern,
                Locale.US
            ).format(Date()) + ".meta.csv"
        )
        fileMeta.createNewFile()
        PrintWriter(fileMeta).use { pw ->
            arrayOf(
                "dataset, ${mlConfiguration.dataset.text}",
                "runner, $runner",
                "optimizer, ${mlConfiguration.optimizer.text}",
                "learning rate, ${mlConfiguration.learningRate.text}",
                "momentum, ${mlConfiguration.momentum?.text ?: "<null>"}",
                "l2, ${mlConfiguration.l2.text}",
                "batchSize, ${mlConfiguration.batchSize.text}",
                "iteratorDistribution, ${mlConfiguration.iteratorDistribution.text}",
                "maxTestSamples, ${mlConfiguration.maxTestSamples.text}",
                "gar, ${mlConfiguration.gar.text}",
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
        listener: EvaluativeListener?,
        model: Model?,
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

    fun skip() {
        dataLines.add(Array(dataLines[0].size) { "-" })

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
