package nl.tudelft.trustchain.fedml.ai.dataset.har

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.dataset.DataSet
import java.io.File


class HARDataSetIterator(
    iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    baseDirectory: File,
    behavior: Behaviors
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    HARDataFetcher(
        baseDirectory,
        seed,
        iteratorConfiguration.distribution,
        dataSetType,
        if (dataSetType == CustomDataSetType.TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        behavior
    )
) {
    override val testBatches by lazy {customFetcher.testBatches}

    override fun getLabels(): List<String> {
        return (fetcher as HARDataFetcher).labels
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors
        ): HARDataSetIterator {
            return HARDataSetIterator(iteratorConfiguration, seed, dataSetType, baseDirectory, behavior)
        }
    }
}

