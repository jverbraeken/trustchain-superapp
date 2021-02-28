package nl.tudelft.trustchain.fedml

import nl.tudelft.trustchain.fedml.ai.*
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.cifar.CustomCifar10DataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.har.HARDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.mnist.CustomMnistDataSetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.mobi_act.MobiActDataSetIterator
import nl.tudelft.trustchain.fedml.ai.gar.*
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020Krum
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.Fang2020TrimmedMean
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.ModelPoisoningAttack
import nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack.NoAttack
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.*
import org.nd4j.linalg.schedule.ISchedule
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import java.io.File


enum class Datasets(
    val id: String,
    val text: String,
    val defaultOptimizer: Optimizers,
    val defaultLearningRate: LearningRates,
    val defaultMomentum: Momentums,
    val defaultL2: L2Regularizations,
    val defaultBatchSize: BatchSizes,
    val defaultIteratorDistribution: IteratorDistributions,
    val architecture: (nnConfiguration: NNConfiguration, seed: Int, mode: NNConfigurationMode) -> MultiLayerConfiguration,
    val inst: (iteratorConfiguration: DatasetIteratorConfiguration, seed: Long, dataSetType: CustomDataSetType, baseDirectory: File, behavior: Behaviors, transfer: Boolean) -> CustomDataSetIterator,
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
        ::generateDefaultMNISTConfiguration,
        CustomMnistDataSetIterator::create,
    ),
    CIFAR10(
        "cifar10",
        "CIFAR-10",
        Optimizers.ADAM,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_5EM3,
        BatchSizes.BATCH_32,
        IteratorDistributions.DISTRIBUTION_CIFAR_50,
        ::generateDefaultCIFARConfiguration,
        CustomCifar10DataSetIterator::create,
    ),

    /*TINYIMAGENET(
        "tinyimagenet",
        "Tiny ImageNet",
        Optimizers.AMSGRAD,
        LearningRates.SCHEDULE2,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_64,
        IteratorDistributions.DISTRIBUTION_MNIST_1,
        Runner::generateDefaultTinyImageNetConfiguration,
        CustomMnistDataSetIterator::create
    ),*/
    HAR(
        "har",
        "HAR",
        Optimizers.ADAM,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_32,
        IteratorDistributions.DISTRIBUTION_HAR_100,
        ::generateDefaultHARConfiguration,
        HARDataSetIterator::create,
    ),
    MOBI_ACT(
        "mobi_act",
        "Mobi Act",
//        Optimizers.NESTEROVS,
        Optimizers.ADAM,
//        LearningRates.SCHEDULE1,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_5,
        IteratorDistributions.DISTRIBUTION_WISDM_100,
        ::generateDefaultMobiActConfiguration,
        MobiActDataSetIterator::create,
    ),
}

private fun createEmnistDataSetIterator(batchSize: Int, train: Boolean): DataSetIterator {
    return EmnistDataSetIterator(EmnistDataSetIterator.Set.LETTERS, batchSize, train)
}

fun loadDataset(dataset: String?) = Datasets.values().firstOrNull { it.id == dataset }


enum class BatchSizes(val id: String, val text: String, val value: Int) {
    BATCH_1("batch_1", "1", 1),
    BATCH_5("batch_5", "5", 5),
    BATCH_16("batch_16", "16", 16),
    BATCH_32("batch_32", "32", 32),
    BATCH_64("batch_64", "64", 64),
    BATCH_96("batch_96", "96", 96),
    BATCH_200("batch_200", "200", 200)
}

fun loadBatchSize(batchSize: String?) = BatchSizes.values().firstOrNull { it.id == batchSize }

enum class IteratorDistributions(val id: String, val text: String, val value: IntArray) {
    DISTRIBUTION_MNIST_1("mnist_100", "MNIST 100", intArrayOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_MNIST_2("mnist_500", "MNIST 500", intArrayOf(500, 500, 500, 500, 500, 500, 500, 500, 500, 500)),
    DISTRIBUTION_MNIST_3(
        "mnist_0_to_7_with_100",
        "MNIST 0 to 7 with 100",
        intArrayOf(100, 100, 100, 100, 100, 100, 100, 0, 0, 0)
    ),
    DISTRIBUTION_MNIST_4(
        "mnist_4_to_10_with_100",
        "MNIST 4 to 10 with 100",
        intArrayOf(0, 0, 0, 0, 100, 100, 100, 100, 100, 100)
    ),
    DISTRIBUTION_MNIST_5("mnist_7_to_4_with_100", "MNIST 0 to 7 with 100", intArrayOf(100, 100, 100, 0, 0, 0, 0, 100, 100, 100)),
    DISTRIBUTION_CIFAR_50("cifar_50", "CIFAR 50", intArrayOf(50, 50, 50, 50, 50, 50, 50, 50, 50, 50)),
    DISTRIBUTION_HAR_100("har_100", "HAR 100", intArrayOf(100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_WISDM_100("wisdm_100", "WISDM 100", intArrayOf(100, 100, 100, 100, 100, 100)),
}

fun loadIteratorDistribution(iteratorDistribution: String?) =
    IteratorDistributions.values().firstOrNull { it.id == iteratorDistribution }

enum class MaxTestSamples(val id: String, val text: String, val value: Int) {
    NUM_10("num_10", "10", 10),
    NUM_20("num_20", "20", 20),
    NUM_40("num_40", "40", 40),
    NUM_100("num_100", "100", 100),
    NUM_200("num_200", "200", 200)
}

fun loadMaxTestSample(maxTestSample: String?) = MaxTestSamples.values().firstOrNull { it.id == maxTestSample }

enum class Optimizers(
    val id: String,
    val text: String,
    val inst: (LearningRates) -> IUpdater,
) {
    NESTEROVS("nesterovs", "Nesterovs", { learningRate -> Nesterovs(learningRate.schedule) }),
    ADAM("adam", "Adam", { learningRate -> Adam(learningRate.schedule) }),
    SGD("sgd", "SGD", { learningRate -> Sgd(learningRate.schedule) }),
    RMSPROP("rmsprop", "RMSprop", { learningRate -> RmsProp(learningRate.schedule) }),
    AMSGRAD("amsgrad", "AMSGRAD", { learningRate -> AMSGrad(learningRate.schedule) })
}

fun loadOptimizer(optimizer: String?) = Optimizers.values().firstOrNull { it.id == optimizer }

enum class LearningRates(val id: String, val text: String, val schedule: ISchedule) {
    RATE_1EM3(
        "rate_1em3",
        "1e-3", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 1e-3))
    ),
    RATE_5EM2(
        "rate_5em2",
        "5e-2", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 0.05))
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
    )
}

fun loadLearningRate(learningRate: String?) = LearningRates.values().firstOrNull { it.id == learningRate }

enum class Momentums(val id: String, val text: String, val value: Double?) {
    NONE("none", "<none>", null),
    MOMENTUM_1EM3("momentum_1em3", "1e-3", 1e-3)
}

fun loadMomentum(momentum: String?) = Momentums.values().firstOrNull { it.id == momentum }

enum class L2Regularizations(val id: String, val text: String, val value: Double) {
    L2_5EM3("l2_5em3", "5e-3", 5e-3),
    L2_1EM4("l2_1em4", "1e-4", 1e-4),
}

fun loadL2Regularization(l2: String?) = L2Regularizations.values().firstOrNull { it.id == l2 }

enum class MaxIterations(val id: String, val text: String, val value: Int) {
    ITER_25("iter_25", "25", 25),
    ITER_50("iter_50", "50", 50),
    ITER_100("iter_100", "100", 100),
    ITER_150("iter_150", "150", 150),
    ITER_200("iter_200", "200", 200),
    ITER_250("iter_250", "250", 250),
    ITER_300("iter_300", "300", 300),
    ITER_400("iter_400", "400", 400),
    ITER_500("iter_500", "500", 500),
    ITER_1000("iter_1000", "1000", 1000),
}

fun loadMaxIteration(iteration: String?) = MaxIterations.values().firstOrNull { it.id == iteration }

// Gradient Aggregation Rule
enum class GARs(
    val id: String,
    val text: String,
    val obj: AggregationRule,
    val defaultModelPoisoningAttack: ModelPoisoningAttacks,
) {
    NONE("none", "None", NoAveraging(), ModelPoisoningAttacks.NONE),
    AVERAGE("average", "Simple average", Average(), ModelPoisoningAttacks.NONE),
    MEDIAN("median", "Median", Median(), ModelPoisoningAttacks.FANG_2020_MEDIAN),
    KRUM("krum", "Krum (b=1)", Krum(1), ModelPoisoningAttacks.FANG_2020_KRUM),
    BRIDGE("bridge", "Bridge (b=1)", Bridge(1), ModelPoisoningAttacks.FANG_2020_TRIMMED_MEAN),
    MOZI("mozi", "Mozi (frac=0.5)", Mozi(0.5), ModelPoisoningAttacks.NONE),
    BRISTLE("bristle", "Bristle", Bristle(), ModelPoisoningAttacks.NONE)
}

fun loadGAR(gar: String?) = GARs.values().firstOrNull { it.id == gar }

enum class CommunicationPatterns(val id: String, val text: String) {
    ALL("all", "All"),
    RANDOM("random", "Random"),
    RR("rr", "Round-robin"),
    RING("ring", "Ring")
}

fun loadCommunicationPattern(communicationPattern: String?) =
    CommunicationPatterns.values().firstOrNull { it.id == communicationPattern }

enum class Behaviors(val id: String, val text: String) {
    BENIGN("benign", "Benign"),
    NOISE("noise", "Noise"),
    LABEL_FLIP_2("label_flip_2", "Label flip 2"),
    LABEL_FLIP_ALL("label_flip_all", "Label flip all")
}

fun loadBehavior(behavior: String?) = Behaviors.values().firstOrNull { it.id == behavior }

enum class Slowdowns(val id: String, val text: String, val multiplier: Double) {
    NONE("none", "-", 1.0),
    D2("d2", "x 0.5", 0.5),
    D5("d5", "x 0.2", 0.2),
}

fun loadSlowdown(slowdown: String?) = Slowdowns.values().firstOrNull { it.id == slowdown }

enum class TransmissionRounds(val id: String, val text: String, val rounds: Int) {
    N0("n0", "0", 0),
    N150("n150", "150", 150)
}

fun loadTransmissionRound(transmissionRound: String?) = TransmissionRounds.values().firstOrNull { it.id == transmissionRound }

enum class ModelPoisoningAttacks(val id: String, val text: String, val obj: ModelPoisoningAttack) {
    NONE("none", "<none>", NoAttack()),
    FANG_2020_TRIMMED_MEAN("fang_2020_trimmed_mean", "Fang 2020 (trimmed mean)", Fang2020TrimmedMean(2)),
    FANG_2020_MEDIAN("fang_2020_median", "Fang 2020 (median)", Fang2020TrimmedMean(2)), // Attack is the same as for mean
    FANG_2020_KRUM("fang_2020_krum", "Fang 2020 (krum)", Fang2020Krum(2))
}

fun loadModelPoisoningAttack(modelPoisoningAttack: String?) =
    ModelPoisoningAttacks.values().firstOrNull { it.id == modelPoisoningAttack }

enum class NumAttackers(val id: String, val text: String, val num: Int) {
    NUM_0("num_0", "0", 0),
    NUM_1("num_1", "1", 1),
    NUM_2("num_2", "2", 2),
    NUM_3("num_3", "3", 3),
    NUM_4("num_4", "4", 4),
    NUM_10("num_10", "10", 10),
    NUM_20("num_20", "20", 20),
    NUM_35("num_35", "35", 35),
    NUM_75("num_75", "75", 75),
    NUM_175("num_175", "175", 175)
}

fun loadNumAttackers(numAttackers: String?) = NumAttackers.values().firstOrNull { it.id == numAttackers }
