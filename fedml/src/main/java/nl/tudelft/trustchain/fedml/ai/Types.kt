package nl.tudelft.trustchain.fedml.ai

import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.nd4j.linalg.schedule.ISchedule
import org.nd4j.linalg.schedule.MapSchedule
import org.nd4j.linalg.schedule.ScheduleType
import kotlin.reflect.KFunction4


enum class Datasets(
    val identifier: String,
    val defaultOptimizer: Optimizers,
    val defaultLearningRate: LearningRates,
    val defaultMomentum: Momentums,
    val defaultL2: L2Regularizations,
    val defaultBatchSize: BatchSizes,
    val defaultArchitecture: KFunction4<Runner, Optimizers, LearningRates, L2Regularizations, MultiLayerConfiguration>
) {
    MNIST(
        "MNIST",
//        Optimizers.NESTEROVS,
        Optimizers.ADAM,
//        LearningRates.SCHEDULE1,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_5EM3,
        BatchSizes.BATCH_64,
        Runner::generateDefaultMNISTConfiguration
    ),
    CIFAR10(
        "CIFAR-10",
        Optimizers.RMSPROP,
        LearningRates.SCHEDULE2,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_64,
        Runner::generateDefaultCIFARConfiguration
    ),
    TINYIMAGENET(
        "Tiny ImageNet",
        Optimizers.AMSGRAD,
        LearningRates.SCHEDULE2,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_64,
        Runner::generateDefaultTinyImageNetConfiguration
    ),
    HAL(
        "HAL",
        Optimizers.ADAM,
        LearningRates.RATE_1EM3,
        Momentums.NONE,
        L2Regularizations.L2_1EM4,
        BatchSizes.BATCH_32,
        Runner::generateDefaultHALConfiguration
    ),
}

enum class Optimizers(
    val identifier: String
) {
    NESTEROVS("Nesterovs"),
    ADAM("Adam"),
    SGD("SGD"),
    RMSPROP("RMSprop"),
    AMSGRAD("AMSGRAD")
}

enum class LearningRates(val identifier: String, val schedule: ISchedule) {
    RATE_1EM3(
        "1e-3", MapSchedule(ScheduleType.ITERATION, hashMapOf(0 to 1e-3))
    ),
    SCHEDULE1(
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

enum class Momentums(val identifier: String, val value: Double?) {
    NONE("<none>", null),
    MOMENTUM_1EM3("1e-3", 1e-3)
}

enum class L2Regularizations(val identifier: String, val value: Double) {
    L2_5EM3("5e-3", 5e-3),
    L2_1EM4("1e-4", 1e-4)
}

enum class BatchSizes(val identifier: String, val value: Int) {
    BATCH_1("1", 1),
    BATCH_32("32", 32),
    BATCH_64("64", 64)
}

enum class Epochs(val identifier: String, val value: Int) {
    EPOCH_1("1", 1),
    EPOCH_5("5", 5),
    EPOCH_25("25", 25),
    EPOCH_50("50", 50)
}
