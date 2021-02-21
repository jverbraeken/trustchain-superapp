package nl.tudelft.trustchain.fedml.ai.gar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import nl.tudelft.trustchain.fedml.d
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

private val logger = KotlinLogging.logger("Average")

class NoAveraging : AggregationRule() {
    override fun integrateParameters(
        network: MultiLayerNetwork,
        oldModel: INDArray,
        gradient: INDArray,
        newOtherModels: Map<Int, INDArray>,
        recentOtherModels: ArrayDeque<Pair<Int, INDArray>>,
        testDataSetIterator: CustomDataSetIterator,
        countPerPeer: Map<Int, Int>,
        logging: Boolean
    ): INDArray {
        logger.d(logging) { formatName("No averaging") }
        return oldModel.sub(gradient)
    }

    override fun isDirectIntegration(): Boolean {
        return false
    }
}
