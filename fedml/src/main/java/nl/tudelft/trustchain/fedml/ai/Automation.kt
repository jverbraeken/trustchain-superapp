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
    val file = Paths.get(baseDirectory.path, "automation_time.json").toFile()
    val string = file.readLines().joinToString("")
    return Json.decodeFromString(string)
}

private val ISOLATED_FIGURE_NAME = arrayOf("Figure 4.3")
private val ISOLATED_FIGURE_GAR = arrayOf("bristle")

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
    val maxIterations = loadMaxIteration(automation.fixedValues.getValue("maxIterations"))!!
    val optimizer = loadOptimizer(automation.fixedValues.getValue("optimizer"))!!
    val learningRate = loadLearningRate(automation.fixedValues.getValue("learningRate"))!!
    val momentum = loadMomentum(automation.fixedValues.getValue("momentum"))!!
    val l2Regularization = loadL2Regularization(automation.fixedValues.getValue("l2Regularization"))!!
    val communicationPattern = loadCommunicationPattern(automation.fixedValues.getValue("communicationPattern"))!!
    val iterationsBeforeEvaluation = automation.fixedValues.getValue("iterationsBeforeEvaluation").toInt()
    val iterationsBeforeSending = automation.fixedValues.getValue("iterationsBeforeSending").toInt()
    val figures = automation.figures
    val myFigures =
        if (automationPart == -1)
            figures.filter { it.name in ISOLATED_FIGURE_NAME }
        else {
            listOf(figures[automationPart])
            /*when (automationPart) {
                0 -> figures.subList(0, 12)
                1 -> figures.subList(12, 14)
                2 -> figures.subList(14, 16)
                3 -> figures.subList(16, 18)
                4 -> figures.subList(18, 20)
                5 -> figures.subList(20, 22)
                6 -> listOf(figures[22], figures[23])
                7 -> listOf(figures[24], figures[25])
                8 -> listOf(figures[26], figures[27])
                9 -> listOf(figures[28])
                10 -> listOf(figures[29], figures[30])
                11 -> listOf(figures[31])
                12 -> listOf(figures[32])
                13 -> listOf(figures[33], figures[34])
                14 -> listOf(figures[35], figures[36])
                15 -> listOf(figures[37], figures[38])
                else -> throw RuntimeException("Impossible")
            }*/
        }
    logger.debug { "myFigures: ${myFigures.map { it.name }}" }

    for (figure in myFigures) {
        configurations.add(arrayListOf())
        figureNames.add(figure.name)
        val dataset = loadDataset(figure.fixedValues.getValue("dataset"))!!
        val overrideMaxIterations = figure.fixedValues["maxIterations"]
        val behavior = loadBehavior(figure.fixedValues.getValue("behavior"))!!
        val modelPoisoningAttack = loadModelPoisoningAttack(figure.fixedValues.getValue("modelPoisoningAttack"))!!
        val numNodes = figure.fixedValues.getValue("numNodes").toInt()
        val numAttackers = loadNumAttackers(figure.fixedValues.getValue("numAttackers"))!!
        val firstNodeSpeed = figure.fixedValues["firstNodeSpeed"]?.toInt() ?: 0
        val firstNodeJoiningLate = figure.fixedValues["firstNodeJoiningLate"]?.equals("true") ?: false
        val overrideIteratorDistribution = figure.iteratorDistributions
        val overrideBatchSize = figure.fixedValues["batchSize"]
        val overrideIteratorDistributionSoft = figure.fixedValues["iteratorDistribution"]
        val overrideCommunicationPattern = figure.fixedValues["communicationPattern"]

        for (test in figure.tests) {
            val gar = loadGAR(test.gar)!!
            if (automationPart == -1 && gar.id !in ISOLATED_FIGURE_GAR) continue


            for (transfer in booleanArrayOf(true/*, false*/)) {
                if (gar == GARs.BRISTLE && !transfer) {
                    // BRISTLE can only work with transfer learning; otherwise all layers except for its outputlayer will stay 0
                    continue
                }
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
                            maxIteration = loadMaxIteration(overrideMaxIterations) ?: maxIterations,
                            gar = gar,
                            communicationPattern = loadCommunicationPattern(overrideCommunicationPattern)
                                ?: communicationPattern,
                            behavior = if (node < numNodes - numAttackers.num) Behaviors.BENIGN else behavior,
                            slowdown = slowdown,
                            joiningLate = if (node == 0 && firstNodeJoiningLate) TransmissionRounds.N150 else TransmissionRounds.N0,
                            iterationsBeforeEvaluation = iterationsBeforeEvaluation,
                            iterationsBeforeSending = iterationsBeforeSending,
                            transfer = transfer
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
    }
    return Pair(configurations, figureNames)
}
