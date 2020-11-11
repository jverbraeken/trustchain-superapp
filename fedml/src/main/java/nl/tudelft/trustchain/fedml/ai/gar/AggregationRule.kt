package nl.tudelft.trustchain.fedml.ai.gar

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util.concurrent.ConcurrentLinkedDeque

abstract class AggregationRule {
    abstract fun integrateParameters(
        oldModel: INDArray,
        gradient: INDArray,
        otherModels: List<INDArray>,
        network: MultiLayerNetwork,
        testDataSetIterator: DataSetIterator,
        allOtherModelsBuffer: ConcurrentLinkedDeque<INDArray>
    ): INDArray

    protected fun formatName(name: String): String {
        return "<====      $name      ====>"
    }
}
