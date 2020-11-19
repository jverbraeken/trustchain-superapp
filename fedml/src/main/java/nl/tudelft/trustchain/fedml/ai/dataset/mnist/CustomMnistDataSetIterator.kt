package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import java.io.File


class CustomMnistDataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: DataSetType,
    behavior: Behaviors
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    CustomMnistDataFetcher(
        iteratorConfiguration.distribution.value,
        seed,
        dataSetType,
        if (dataSetType == DataSetType.TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        behavior
    )
) {
    override val testBatches by lazy { customFetcher.testBatches }

    override fun getLabels(): List<String> {
        return (fetcher as CustomMnistDataFetcher).labels
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: DataSetType,
            baseDirectory: File,
            behavior: Behaviors
        ): CustomMnistDataSetIterator {
            return CustomMnistDataSetIterator(iteratorConfiguration, seed, dataSetType, behavior)
        }
    }
}
