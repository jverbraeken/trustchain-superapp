package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.dataset.api.DataSet
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
        iteratorConfiguration.maxTestSamples?.value ?: Integer.MAX_VALUE,
        behavior
    )
) {
    override val testBatches: List<DataSet> by lazy {customFetcher.testBatches}

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
