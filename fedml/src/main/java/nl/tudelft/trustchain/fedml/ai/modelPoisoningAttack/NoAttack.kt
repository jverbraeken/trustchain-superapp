package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
import kotlin.random.Random

private val logger = KotlinLogging.logger("NoAttack")

class NoAttack : ModelPoisoningAttack() {
    override fun generateAttack(
        numAttackers: NumAttackers,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray> {
        return HashMap()
    }
}
