package nl.tudelft.trustchain.fedml.ai.dataset.mnist

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.deeplearning4j.datasets.fetchers.DataSetType
import java.io.File


class CustomMnistDataSetIterator(
    val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomBaseDatasetIterator(
    iteratorConfiguration.batchSize.value,
    -1,
    CustomMnistDataFetcher(
        iteratorConfiguration.distribution.toIntArray(),
        seed,
        dataSetType,
        if (dataSetType == CustomDataSetType.TRAIN) Integer.MAX_VALUE else iteratorConfiguration.maxTestSamples.value,
        behavior,
        transfer
    )
), CustomDataSetIterator {
    override val testBatches by lazy { customFetcher.testBatches }

    override fun getLabels(): List<String> {
        return iteratorConfiguration.distribution
            .zip(iteratorConfiguration.distribution.indices)
            .filter { (numSamples, _) -> numSamples > 0 }
            .map { it.second.toString() }
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors,
            transfer: Boolean
        ): CustomMnistDataSetIterator {
            return CustomMnistDataSetIterator(iteratorConfiguration, seed, dataSetType, behavior, transfer)
        }
    }
}
