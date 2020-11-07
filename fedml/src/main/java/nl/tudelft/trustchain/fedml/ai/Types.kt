package nl.tudelft.trustchain.fedml.ai

import nl.tudelft.trustchain.fedml.ai.gar.*
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
    HAL(
        "hal",
        "HAL",
        Optimizers.ADAM,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_32,
        IteratorDistributions.DISTRIBUTION_HAR_1,
        Runner::generateDefaultHALConfiguration
    ),
}

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

enum class LearningRates(val id: String, val text: String, val schedule: ISchedule) {
    RATE_1EM3("rate_1em3",
        "1e-3", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 1e-3))
    ),
    SCHEDULE1("schedule1",
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
    SCHEDULE2("schedule2",
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

enum class Momentums(val id: String, val text: String, val value: Double?) {
    NONE("none", "<none>", null),
    MOMENTUM_1EM3("momentum_1em3", "1e-3", 1e-3)
}

enum class L2Regularizations(val id: String, val text: String, val value: Double) {
    L2_5EM3("l2_5em3", "5e-3", 5e-3),
    L2_1EM4("l2_1em4", "1e-4", 1e-4)
}

enum class BatchSizes(val id: String, val text: String, val value: Int) {
    BATCH_1("batch_1", "1", 1),
    BATCH_5("batch_5", "5", 5),
    BATCH_32("batch_32", "32", 32),
    BATCH_64("batch_64", "64", 64)
}

enum class Epochs(val id: String, val text: String, val value: Int) {
    EPOCH_1("epoch_1", "1", 1),
    EPOCH_5("epoch_5", "5", 5),
    EPOCH_25("epoch_25", "25", 25),
    EPOCH_50("epoch_50", "50", 50)
}

enum class IteratorDistributions(val id: String, val text: String, val value: List<Int>) {
    DISTRIBUTION_MNIST_1("mnist_1", "MNIST 100", arrayListOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_MNIST_2("mnist_2", "MNIST 500", arrayListOf(500, 500, 500, 500, 500, 500, 500, 500, 500, 500)),
    DISTRIBUTION_CIFAR_1("cifar_1", "CIFAR 1", arrayListOf(100, 100, 100, 100, 100, 100, 100, 100, 100, 100)),
    DISTRIBUTION_HAR_1("har_1", "HAR 1", arrayListOf(100, 100, 100, 100, 100, 100)),
}

enum class MaxSamples(val id: String, val text: String, val value: Int) {
    NUM_200("num_200", "200", 200)
}

// Gradient Aggregation Rule
enum class GARs(val id: String, val text: String, val obj: AggregationRule) {
    AVERAGE("average", "Simple average", Average()),
    MEDIAN("median", "Median", Median()),
    CWTRIMMEDMEAN("cwtrimmedmean", "Trimmed Mean (b=1)", CWTrimmedMean(1)),
    KRUM("krum", "Krum (b=1)", Krum(1)),
    BRIDGE("bridge", "Bridge (b=1)", Bridge(1)),
    MOZI("mozi", "Mozi (frac=0.5)", Mozi(0.5))
}

enum class CommunicationPatterns(val id: String, val text: String) {
    ALL("all", "All"),
    RANDOM("random", "Random"),
    RR("rr", "Round-robin"),
    RING("ring", "Ring")
}

enum class Behaviors(val id: String, val text: String) {
    BENIGN("benign", "Benign"),
    NOISE("noise", "Noise"),
    LABEL_FLIP("label_flip", "Label flip")
}
