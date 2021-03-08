package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.Behaviors
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import nl.tudelft.trustchain.fedml.ai.DatasetIteratorConfiguration
import nl.tudelft.trustchain.fedml.ai.dataset.CustomDataSetIterator
import org.datavec.image.transform.FlipImageTransform
import org.datavec.image.transform.ImageTransform
import org.datavec.image.transform.PipelineImageTransform
import org.datavec.image.transform.RotateImageTransform
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import java.io.File
import kotlin.random.Random

private val logger = KotlinLogging.logger("CustomCifar10DataSetIterator")

class CustomCifar10DataSetIterator(
    val iteratorConfiguration: DatasetIteratorConfiguration,
    seed: Long,
    dataSetType: CustomDataSetType,
    behavior: Behaviors,
    transfer: Boolean,
) : CustomRecordReaderDataSetIterator(
    CustomCifar10Fetcher().getRecordReader(
        seed,
        null,
        dataSetType,
        CustomPipelineImageTransform(CustomFlipImageTransform(0, java.util.Random(42)), RotateImageTransform(java.util.Random(42), 0.0f, 0.0f, 20.0f, 0.1f)),
        iteratorConfiguration.distribution.toIntArray(),
        if (dataSetType == CustomDataSetType.TEST || dataSetType == CustomDataSetType.FULL_TEST) iteratorConfiguration.maxTestSamples.value else Integer.MAX_VALUE,
        transfer,
    ),
    iteratorConfiguration.batchSize.value,
    1,
    if (transfer) CustomCifar10Fetcher.NUM_LABELS_TRANSFER else CustomCifar10Fetcher.NUM_LABELS_REGULAR
), CustomDataSetIterator {
    override val testBatches by lazy { recordReader.testBatches!! }

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
            transfer: Boolean,
        ): CustomCifar10DataSetIterator {
            val iterator = CustomCifar10DataSetIterator(iteratorConfiguration, seed, dataSetType, behavior, transfer)
            iterator.setPreProcessor(ImagePreProcessingScaler(0.0, 1.0))
            return iterator
        }
    }
}
