package nl.tudelft.trustchain.fedml.ai.modelPoisoningAttack

import nl.tudelft.trustchain.fedml.NumAttackers
import org.nd4j.linalg.api.ndarray.INDArray
import java.util.ArrayList
import kotlin.random.Random

abstract class ModelPoisoningAttack {
    abstract fun generateAttack(
        numAttackers: NumAttackers,
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: Map<Int, INDArray>,
        random: Random
    ): Map<Int, INDArray>

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }

    protected fun transformToResult(newModels: ArrayList<INDArray>): Map<Int, INDArray> {
        var attackNum = -1
        return newModels.map {
            attackNum--
            Pair(attackNum, it)
        }.toMap()
    }
}
