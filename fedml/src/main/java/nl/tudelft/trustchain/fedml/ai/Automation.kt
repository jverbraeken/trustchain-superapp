package nl.tudelft.trustchain.fedml.ai

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json
import nl.tudelft.trustchain.fedml.*
import java.io.File
import java.nio.file.Paths

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

private const val ISOLATED_FIGURE_NAME = "Figure 2.2"
private const val ISOLATED_FIGURE_GAR = "krum"

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
    val iteratorDistribution_ = automation.fixedValues.getValue("iteratorDistribution")
    val iteratorDistribution = if (iteratorDistribution_.startsWith('[')) {
        iteratorDistribution_
            .substring(1, iteratorDistribution_.length - 1)
            .split(", ")
            .map { it.toInt() }
    } else {
        loadIteratorDistribution(iteratorDistribution_)!!.value
    }
    val maxTestSample = loadMaxTestSample(automation.fixedValues.getValue("maxTestSample"))!!
    val optimizer = loadOptimizer(automation.fixedValues.getValue("optimizer"))!!
    val learningRate = loadLearningRate(automation.fixedValues.getValue("learningRate"))!!
    val momentum = loadMomentum(automation.fixedValues.getValue("momentum"))!!
    val l2Regularization = loadL2Regularization(automation.fixedValues.getValue("l2Regularization"))!!
    val communicationPattern = loadCommunicationPattern(automation.fixedValues.getValue("communicationPattern"))!!
    val figures = automation.figures
    val myFigures =
        if (automationPart == -1)
            figures.filter { it.name == ISOLATED_FIGURE_NAME }
        else
            figures.subList(automationPart * (figures.size / 4), (automationPart + 1) * (figures.size / 4))

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
        val overrideIteratorDistribution_ = figure.iteratorDistributions

        for (test in figure.tests) {
            val gar = loadGAR(test.gar)!!
            if (automationPart == -1 && gar.id != ISOLATED_FIGURE_GAR) continue

            configurations.last().add(arrayListOf())

            for (node in 0 until numNodes) {
                val overrideIteratorDistributionForNode =
                    overrideIteratorDistribution_?.get(node % overrideIteratorDistribution_.size)
                val distribution = if (overrideIteratorDistributionForNode != null) {
                    if (overrideIteratorDistributionForNode.startsWith('[')) {
                        overrideIteratorDistributionForNode
                            .substring(1, overrideIteratorDistributionForNode.length - 1)
                            .split(", ")
                            .map { it.toInt() }
                    } else {
                        loadIteratorDistribution(overrideIteratorDistributionForNode)!!.value
                    }
                } else {
                    iteratorDistribution
                }
                val configuration = MLConfiguration(
                    dataset,
                    DatasetIteratorConfiguration(
                        batchSize = batchSize,
                        maxTestSamples = maxTestSample,
                        distribution = distribution
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
                        behavior = behavior,
                        slowdown = if ((node == 0 && firstNodeSpeed == -1) || (node != 0 && firstNodeSpeed == 1)) Slowdowns.D2 else Slowdowns.NONE,
                        joiningLate = if (node == 0 && firstNodeJoiningLate) TransmissionRounds.N2 else TransmissionRounds.N0
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
