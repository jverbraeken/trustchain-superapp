package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

abstract class AggregationRule {
    private val mpl = KotlinLogging.logger("AggregationRule")

    abstract fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomBaseDatasetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean,
    ): INDArray

    abstract fun isDirectIntegration(): Boolean

    protected fun debug(logging: Boolean, msg: () -> Any?) {
        if (logging) mpl.debug(msg)
    }

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }
}
