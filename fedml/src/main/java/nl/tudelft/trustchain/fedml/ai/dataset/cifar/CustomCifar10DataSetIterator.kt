package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomBaseDatasetIterator
import nl.tudelft.trustchain.fedml.ai.dataset.mnist.CustomMnistDataFetcher
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.linalg.dataset.DataSet
import java.io.File


class CustomCifar10DataSetIterator(
    private val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    behavior: Behaviors
) : CustomBaseDatasetIterator(
    1,
    1,
    CustomMnistDataFetcher(arrayListOf(), 0L, CustomDataSetType.FULL_TEST, 0, Behaviors.BENIGN)
    /*CustomCifar10Fetcher(
        iteratorConfiguration.distribution,
        if (dataSetType == CustomDataSetType.TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        behavior
    ).getRecordReader(seed, null, *//*dataSetType*//*DataSetType.TRAIN, null), // DataSetType.TRAIN I just put there to not having to fix the code right away
    iteratorConfiguration.batchSize.value,
    1,
    Cifar10Fetcher.NUM_LABELS*/
) {
    override val testBatches: List<DataSet?>
        get() = arrayListOf() // complete nonsense

    override fun getLabels(): List<String> {
        return iteratorConfiguration
            .distribution
            .filter { it > 0 }
            .map { it.toString() }
            .toCollection(ArrayList())
    }

    companion object {
        fun create(
            iteratorConfiguration: DatasetIteratorConfiguration,
            seed: Long,
            dataSetType: CustomDataSetType,
            baseDirectory: File,
            behavior: Behaviors
        ): CustomCifar10DataSetIterator {
            return CustomCifar10DataSetIterator(iteratorConfiguration, seed, dataSetType, behavior)
        }
    }
}
