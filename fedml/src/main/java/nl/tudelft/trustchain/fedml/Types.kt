package nl.tudelft.trustchain.fedml

import nl.tudelft.trustchain.fedml.ai.NNConfiguration
import nl.tudelft.trustchain.fedml.ai.Runner
import nl.tudelft.trustchain.fedml.ai.gar.*
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020Krum
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020TrimmedMean
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.ModelPoisoningAttack
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.NoAttack
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.schedule.ISchedule
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import kotlin.reflect.KFunction3


enum class Datasets(
    val id: String,
    val text: String,
    val defaultOptimizer: Optimizers,
    val defaultLearningRate: LearningRates,
    val defaultMomentum: Momentums,
    val defaultL2: L2Regularizations,
    val defaultBatchSize: BatchSizes,
    val defaultIteratorDistribution: IteratorDistributions,
    val architecture: KFunction3<Runner, NNConfiguration, Int, MultiLayerConfiguration>
) {
    MNIST(
        "mnist",
        "MNIST",
//        Optimizers.NESTEROVS,
        Optimizers.ADAM,
//        LearningRates.SCHEDULE1,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_5EM3,
        BatchSizes.BATCH_5,
        IteratorDistributions.DISTRIBUTION_MNIST_2,
        Runner::generateDefaultMNISTConfiguration
    ),
    CIFAR10(
        "cifar10",
        "CIFAR-10",
        Optimizers.RMSPROP,
        LearningRates.SCHEDULE2,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_64,
        IteratorDistributions.DISTRIBUTION_CIFAR_1,
        Runner::generateDefaultCIFARConfiguration
    ),
    TINYIMAGENET(
        "tinyimagenet",
        "Tiny ImageNet",
        Optimizers.AMSGRAD,
        LearningRates.SCHEDULE2,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_64,
        IteratorDistributions.DISTRIBUTION_MNIST_1,
        Runner::generateDefaultTinyImageNetConfiguration
    ),
    HAR(
        "har",
        "HAR",
        Optimizers.ADAM,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_32,
        IteratorDistributions.DISTRIBUTION_HAR_1,
        Runner::generateDefaultHARConfiguration
    ),
}

fun loadDataset(dataset: String) = Datasets.values().first { it.id == dataset }


enum class BatchSizes(val id: String, val text: String, val value: Int) {
    BATCH_1("batch_1", "1", 1),
    BATCH_5("batch_5", "5", 5),
    BATCH_32("batch_32", "32", 32),
    BATCH_64("batch_64", "64", 64)
}

fun loadBatchSize(batchSize: String) = BatchSizes.values().first { it.id == batchSize }

enum class IteratorDistributions(val id: String, val text: String, val value: List<Int>) {
    DISTRIBUTION_MNIST_1("mnist_100", "MNIST 100", arrayListOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_MNIST_2("mnist_500", "MNIST 500", arrayListOf(500, 500, 500, 500, 500, 500, 500, 500, 500, 500)),
    DISTRIBUTION_MNIST_3("mnist_0_to_7_with_100", "MNIST 0 to 7 with 100", arrayListOf(100, 100, 100, 100, 100, 100, 100, 0, 0, 0)),
    DISTRIBUTION_MNIST_4("mnist_4_to_10_with_100", "MNIST 4 to 10 with 100", arrayListOf(0, 0, 0, 0, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_MNIST_5("mnist_7_to_4_with_100", "MNIST 0 to 7 with 100", arrayListOf(100, 100, 100, 0, 0, 0, 0, 100, 100, 100)),
    DISTRIBUTION_CIFAR_1("cifar_1", "CIFAR 1", arrayListOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_HAR_1("har_1", "HAR 1", arrayListOf(100, 100, 100, 100, 100, 100)),
}

fun loadIteratorDistribution(iteratorDistribution: String) =
    IteratorDistributions.values().first { it.id == iteratorDistribution }

enum class MaxTestSamples(val id: String, val text: String, val value: Int) {
    NUM_40("num_40", "40", 40),
    NUM_200("num_200", "200", 200)
}

fun loadMaxTestSample(maxTestSample: String) = MaxTestSamples.values().first { it.id == maxTestSample }

enum class Optimizers(
    val id: String,
    val text: String
) {
    NESTEROVS("nesterovs", "Nesterovs"),
    ADAM("adam", "Adam"),
    SGD("sgd", "SGD"),
    RMSPROP("rmsprop", "RMSprop"),
    AMSGRAD("amsgrad", "AMSGRAD")
}

fun loadOptimizer(optimizer: String) = Optimizers.values().first { it.id == optimizer }

enum class LearningRates(val id: String, val text: String, val schedule: ISchedule) {
    RATE_1EM3(
        "rate_1em3",
        "1e-3", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 1e-3))
    ),
    SCHEDULE1(
        "schedule1",
        "{0 -> 0.06|100 -> 0.05|200 -> 0.028|300 -> 0.006|400 -> 0.001", MapSchedule(
            ScheduleType.ITERATION, hashMapOf(
                0 to 0.06,
                100 to 0.05,
                200 to 0.028,
                300 to 0.006,
                400 to 0.001
            )
        )
    ),
    SCHEDULE2(
        "schedule2",
        "{0 -> 0.001|75 -> 0.005|100 -> 0.003}", MapSchedule(
            ScheduleType.EPOCH,
            hashMapOf(
                0 to 0.01,
                75 to 0.005,
                100 to 0.003
            )
        )
    )
}

fun loadLearningRate(learningRate: String) = LearningRates.values().first { it.id == learningRate }

enum class Momentums(val id: String, val text: String, val value: Double?) {
    NONE("none", "<none>", null),
    MOMENTUM_1EM3("momentum_1em3", "1e-3", 1e-3)
}

fun loadMomentum(momentum: String) = Momentums.values().first { it.id == momentum }

enum class L2Regularizations(val id: String, val text: String, val value: Double) {
    L2_5EM3("l2_5em3", "5e-3", 5e-3),
    L2_1EM4("l2_1em4", "1e-4", 1e-4)
}

fun loadL2Regularization(l2: String) = L2Regularizations.values().first { it.id == l2 }

enum class Epochs(val id: String, val text: String, val value: Int) {
    EPOCH_1("epoch_1", "1", 1),
    EPOCH_5("epoch_5", "5", 5),
    EPOCH_25("epoch_25", "25", 25),
    EPOCH_50("epoch_50", "50", 50)
}

fun loadEpoch(epoch: String) = Epochs.values().first { it.id == epoch }

// Gradient Aggregation Rule
enum class GARs(
    val id: String,
    val text: String,
    val obj: AggregationRule,
    val defaultModelPoisoningAttack: ModelPoisoningAttacks
) {
    AVERAGE("average", "Simple average", Average(), ModelPoisoningAttacks.NONE),
    MEDIAN("median", "Median", Median(), ModelPoisoningAttacks.FANG_2020_MEDIAN),
    CWTRIMMEDMEAN(
        "cwtrimmedmean", "Trimmed Mean (b=1)", CWTrimmedMean(1),
        ModelPoisoningAttacks.FANG_2020_TRIMMED_MEAN
    ),
    KRUM("krum", "Krum (b=1)", Krum(1), ModelPoisoningAttacks.FANG_2020_KRUM),
    BRIDGE("bridge", "Bridge (b=1)", Bridge(1), ModelPoisoningAttacks.NONE),
    MOZI("mozi", "Mozi (frac=0.5)", Mozi(0.5), ModelPoisoningAttacks.NONE),
    BRISTLE(
        "bristle", "Bristle (frac=0.5)", Bristle(0.5),
        ModelPoisoningAttacks.NONE
    )
}

fun loadGAR(gar: String) = GARs.values().first { it.id == gar }

enum class CommunicationPatterns(val id: String, val text: String) {
    ALL("all", "All"),
    RANDOM("random", "Random"),
    RR("rr", "Round-robin"),
    RING("ring", "Ring")
}

fun loadCommunicationPattern(communicationPattern: String) =
    CommunicationPatterns.values().first { it.id == communicationPattern }

enum class Behaviors(val id: String, val text: String) {
    BENIGN("benign", "Benign"),
    NOISE("noise", "Noise"),
    LABEL_FLIP("label_flip", "Label flip")
}

fun loadBehavior(behavior: String) = Behaviors.values().first { it.id == behavior }

enum class ModelPoisoningAttacks(val id: String, val text: String, val obj: ModelPoisoningAttack) {
    NONE("none", "<none>", NoAttack()),
    FANG_2020_TRIMMED_MEAN("fang_2020_trimmed_mean", "Fang 2020, trimmed mean", Fang2020TrimmedMean(2)),
    FANG_2020_MEDIAN("fang_2020_median", "Fang 2020, median", Fang2020TrimmedMean(2)), // Attack is the same as for mean
    FANG_2020_KRUM("fang_2020_krum", "Fang 2020, krum", Fang2020Krum(2))
}

fun loadModelPoisoningAttack(modelPoisoningAttack: String) =
    ModelPoisoningAttacks.values().first { it.id == modelPoisoningAttack }

enum class NumAttackers(val id: String, val text: String, val num: Int) {
    NUM_1("num_1", "1", 1),
    NUM_2("num_2", "2", 2),
    NUM_3("num_3", "3", 3),
    NUM_4("num_4", "4", 4)
}

fun loadNumAttackers(numAttackers: String) = NumAttackers.values().first { it.id == numAttackers }
