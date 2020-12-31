package nl.tudelft.trustchain.fedml.ai.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator

abstract class CustomBaseDatasetIterator(batch: Int, numExamples: Int, protected val customFetcher: CustomBaseDataFetcher) :
    BaseDatasetIterator(batch, numExamples, customFetcher) {
    abstract val testBatches: Array<DataSet?>

    companion object {
        private const val serialVersionUID = -11663679242619894L
    }
}
