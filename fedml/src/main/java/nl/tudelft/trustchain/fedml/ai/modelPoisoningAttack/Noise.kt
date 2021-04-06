package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import kotlin.random.Random

private val logger = KotlinLogging.logger("Noise")

class Noise : ModelPoisoningAttack() {
    override fun generateAttack(
        numAttackers: NumAttackers,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray> {
        logger.debug { formatName("Noise") }
        val numColumns = oldModel.columns()
        val halfNumColumns = numColumns / 2
        val newModels =
            Array<INDArray>(numAttackers.num) { NDArray(Array(oldModel.rows()) { FloatArray(numColumns) { random.nextFloat() / 2 + (if (it < halfNumColumns) -0.2f else 0.2f) } }) }
        return transformToResult(newModels)
    }
}
