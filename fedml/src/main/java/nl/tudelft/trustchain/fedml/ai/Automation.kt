package nl.tudelft.trustchain.fedml.ai

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.*
import java.io.File
import java.nio.file.Paths

private val logger = KotlinLogging.logger("Automation")

@Serializable
data class Automation(val fixedValues: Map<String, String>, val figures: List<Figure>)

@Serializable
data class Figure(
    val name: String,
    val fixedValues: Map<String, String>,
    val tests: List<Test>,
    val iteratorDistributions: List<String>? = null,
)

@Serializable
data class Test(val gar: String)

fun loadAutomation(baseDirectory: File): Automation {
    val file = Paths.get(baseDirectory.path, "automation.config").toFile()
    val string = file.readLines().joinToString("")
    return Json.decodeFromString(string)
}

private const val ISOLATED_FIGURE_NAME = "Figure 3.1"
private val ISOLATED_FIGURE_GAR = arrayOf("average", "bristle")

/**
 * @return 1. the configuration per node, per test, per figure ; 2. the names of the figures
 */
fun generateConfigs(
    automation: Automation,
    automationPart: Int,
): Pair<List<List<List<MLConfiguration>>>, List<String>> {
    val configurations = arrayListOf<MutableList<MutableList<MLConfiguration>>>()
    val figureNames = arrayListOf<String>()

    // global fixed values
    val batchSize = loadBatchSize(automation.fixedValues.getValue("batchSize"))!!
    val iteratorDistribution = automation.fixedValues.getValue("iteratorDistribution")
    val maxTestSample = loadMaxTestSample(automation.fixedValues.getValue("maxTestSample"))!!
    val optimizer = loadOptimizer(automation.fixedValues.getValue("optimizer"))!!
    val learningRate = loadLearningRate(automation.fixedValues.getValue("learningRate"))!!
    val momentum = loadMomentum(automation.fixedValues.getValue("momentum"))!!
    val l2Regularization = loadL2Regularization(automation.fixedValues.getValue("l2Regularization"))!!
    val communicationPattern = loadCommunicationPattern(automation.fixedValues.getValue("communicationPattern"))!!
    val iterationsBeforeEvaluation = automation.fixedValues.getValue("iterationsBeforeEvaluation").toInt()
    val iterationsBeforeSending = automation.fixedValues.getValue("iterationsBeforeSending").toInt()
    val figures = automation.figures
    logger.debug { "Testing figure: ${automationPart * (figures.size / 13)}" }
    val myFigures =
        if (automationPart == -1)
            figures.filter { it.name == ISOLATED_FIGURE_NAME }
        else
            figures.subList(automationPart * (figures.size / 13), (automationPart + 1) * (figures.size / 13))

    for (figure in myFigures) {
        configurations.add(arrayListOf())
        figureNames.add(figure.name)
        val dataset = loadDataset(figure.fixedValues.getValue("dataset"))!!
        val maxIterations = loadMaxIteration(figure.fixedValues.getValue("maxIterations"))!!
        val behavior = loadBehavior(figure.fixedValues.getValue("behavior"))!!
        val modelPoisoningAttack = loadModelPoisoningAttack(figure.fixedValues.getValue("modelPoisoningAttack"))!!
        val numNodes = figure.fixedValues.getValue("numNodes").toInt()
        val numAttackers = loadNumAttackers(figure.fixedValues.getValue("numAttackers"))!!
        val firstNodeSpeed = figure.fixedValues["firstNodeSpeed"]?.toInt() ?: 0
        val firstNodeJoiningLate = figure.fixedValues["firstNodeJoiningLate"]?.equals("true") ?: false
        val overrideIteratorDistribution = figure.iteratorDistributions
        val overrideBatchSize = figure.fixedValues["batchSize"]
        val overrideIteratorDistributionSoft = figure.fixedValues["iteratorDistribution"]

        for (test in figure.tests) {
            val gar = loadGAR(test.gar)!!
            if (automationPart == -1 && gar.id !in ISOLATED_FIGURE_GAR) continue

            configurations.last().add(arrayListOf())

            for (node in 0 until numNodes) {
                val overrideIteratorDistributionForNode =
                    overrideIteratorDistribution?.get(node % overrideIteratorDistribution.size)
                val distribution = if (overrideIteratorDistributionForNode != null) {
                    if (overrideIteratorDistributionForNode.startsWith('[')) {
                        overrideIteratorDistributionForNode
                            .substring(1, overrideIteratorDistributionForNode.length - 1)
                            .split(", ")
                            .map { it.toInt() }
                            .toIntArray()
                    } else {
                        loadIteratorDistribution(overrideIteratorDistributionForNode)!!.value
                    }
                } else {
                    val d = overrideIteratorDistributionSoft ?: iteratorDistribution
                    if (d.startsWith('[')) {
                        d.substring(1, d.length - 1)
                            .split(", ")
                            .map { it.toInt() }
                            .toIntArray()
                    } else {
                        loadIteratorDistribution(d)!!.value
                    }
                }
                val slowdown =
                    if ((node == 0 && firstNodeSpeed == -2) || (node != 0 && firstNodeSpeed == 2)) Slowdowns.D2
                    else if ((node == 0 && firstNodeSpeed == -5) || (node != 0 && firstNodeSpeed == 5)) Slowdowns.D5
                    else Slowdowns.NONE
                val configuration = MLConfiguration(
                    dataset,
                    DatasetIteratorConfiguration(
                        batchSize = loadBatchSize(overrideBatchSize) ?: batchSize,
                        maxTestSamples = maxTestSample,
                        distribution = distribution.toList()
                    ),
                    NNConfiguration(
                        optimizer = optimizer,
                        learningRate = learningRate,
                        momentum = momentum,
                        l2 = l2Regularization
                    ),
                    TrainConfiguration(
                        maxIteration = maxIterations,
                        gar = gar,
                        communicationPattern = communicationPattern,
                        behavior = if (node >= numAttackers.num) Behaviors.BENIGN else behavior,
                        slowdown = slowdown,
                        joiningLate = if (node == 0 && firstNodeJoiningLate) TransmissionRounds.N150 else TransmissionRounds.N0,
                        iterationsBeforeEvaluation = iterationsBeforeEvaluation,
                        iterationsBeforeSending = iterationsBeforeSending
                    ),
                    ModelPoisoningConfiguration(
                        attack = modelPoisoningAttack,
                        numAttackers = numAttackers
                    )
                )
                configurations.last().last().add(configuration)
            }
        }
    }
    return Pair(configurations, figureNames)
}
