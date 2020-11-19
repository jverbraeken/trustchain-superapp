package nl.tudelft.trustchain.fedml.ai.dataset

import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher

abstract class CustomBaseDataFetcher : BaseDataFetcher() {
    abstract val testBatches: List<DataSet?>
}
