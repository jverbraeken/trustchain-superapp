package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.random.Random

abstract class ModelPoisoningAttack {
    abstract fun generateAttack(
        numAttackers: NumAttackers,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        random: Random
    ): Collection<INDArray>

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }
}
