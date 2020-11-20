package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import nl.tudelft.trustchain.fedml.Behaviors
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.ImageTransform
import org.deeplearning4j.datasets.fetchers.Cifar10Fetcher
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.base.Preconditions
import java.io.File
import java.util.*

class CustomCifar10Fetcher(
    private val iteratorDistribution: List<Int>,
    private val maxTestSamples: Int,
    private val behavior: Behaviors
) : Cifar10Fetcher() {
    override fun getRecordReader(
        rngSeed: Long,
        imgDim: IntArray?,
        dataSetType: DataSetType,
        imageTransform: ImageTransform?
    ): RecordReader {
        Preconditions.checkState(
            imgDim == null || imgDim.size == 2,
            "Invalid image dimensions: must be null or length 2. Got: $imgDim"
        )
        // check empty cache
        val localCache = localCacheDir
        deleteIfEmpty(localCache)
        try {
            if (!localCache.exists()) {
                downloadAndExtract()
            }
        } catch (e: Exception) {
            throw RuntimeException("Could not download CIFAR-10", e)
        }
        val rng = Random(rngSeed)
        val datasetPath: File
        datasetPath = when (dataSetType) {
            DataSetType.TRAIN -> File(localCache, "/train/")
            DataSetType.TEST -> File(localCache, "/test/")
            else -> throw IllegalArgumentException("You will need to manually create and iterate a validation directory, CIFAR-10 does not provide labels")
        }

        // dataSetType up file paths
        val pathFilter = RandomPathFilter(rng, *BaseImageLoader.ALLOWED_FORMATS)
        val filesInDir = FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng)
        val filesInDirSplit = filesInDir.sample(pathFilter, *iteratorDistribution.map { it.toDouble() }.toDoubleArray())
        val h = imgDim?.get(0) ?: INPUT_HEIGHT
        val w = imgDim?.get(1) ?: INPUT_WIDTH
        val rr = ImageRecordReader(h.toLong(), w.toLong(), INPUT_CHANNELS.toLong(), ParentPathLabelGenerator(), imageTransform)
        try {
            rr.initialize(filesInDirSplit[0])
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
        return rr
    }
}
