package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator


class CustomMnistDataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Int,
    dataSetType: DataSetType,
    behavior: Behaviors
) : BaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    CustomMnistDataFetcher(
        iteratorConfiguration.distribution.value,
        seed,
        dataSetType,
        iteratorConfiguration.maxTestSamples?.value ?: Integer.MAX_VALUE,
        behavior
    )
) {
    override fun getLabels(): List<String> {
        return (fetcher as CustomMnistDataFetcher).labels
    }
}
