package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import org.apache.commons.io.FileUtils
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.split.CollectionInputSplit
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.transform.ImageTransform
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.deeplearning4j.datasets.fetchers.DataSetType
import org.nd4j.base.Preconditions
import org.nd4j.util.ArchiveUtils
import java.io.File
import java.io.IOException
import java.net.URI
import java.net.URL
import java.util.*
import java.util.zip.Adler32
import java.util.zip.Checksum
import kotlin.collections.HashMap
import kotlin.math.min

private val logger = KotlinLogging.logger("CustomCifar10Fetcher")

class CustomCifar10Fetcher {
    fun dataSetName(set: DataSetType): String {
        return ""
    }

    fun downloadAndExtract() {
        downloadAndExtract(DataSetType.TRAIN)
    }

    /**
     * Downloads and extracts the local dataset.
     *
     * @throws IOException
     */
    fun downloadAndExtract(set: DataSetType?) {
        val localFilename = File(remoteDataUrl(set!!)).name
        val tmpFile = File(System.getProperty("java.io.tmpdir"), localFilename)
        val localCacheDir = getLocalCacheDir()

        // Check empty cache
        if (localCacheDir.exists()) {
            val list = localCacheDir.listFiles()
            if (list == null || list.size == 0) localCacheDir.delete()
        }
        val localDestinationDir = File(localCacheDir, dataSetName(set))
        if (!localDestinationDir.exists()) {
            localCacheDir.mkdirs()
            tmpFile.delete()
            logger.info("Downloading dataset to " + tmpFile.absolutePath)
            FileUtils.copyURLToFile(URL(remoteDataUrl(set)), tmpFile)
        } else {
            // Directory exists and is non-empty - assume OK
            logger.info("Using cached dataset at " + localCacheDir.absolutePath)
            return
        }
        if (expectedChecksum(set) != 0L) {
            logger.info("Verifying download...")
            val adler: Checksum = Adler32()
            FileUtils.checksum(tmpFile, adler)
            val localChecksum = adler.value
            logger.info("Checksum local is " + localChecksum + ", expecting " + expectedChecksum(set))
            if (expectedChecksum(set) != localChecksum) {
                logger.error("Checksums do not match. Cleaning up files and failing...")
                tmpFile.delete()
                throw IllegalStateException("Dataset file failed checksum: $tmpFile - expected checksum " + expectedChecksum(
                    set)
                    + " vs. actual checksum " + localChecksum + ". If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.")
            }
        }
        try {
            ArchiveUtils.unzipFileTo(tmpFile.absolutePath, localCacheDir.absolutePath)
        } catch (t: Throwable) {
            // Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if (localCacheDir.exists()) FileUtils.deleteDirectory(localCacheDir)
            throw t
        }
    }

    private fun getLocalCacheDir(): File {
        return DL4JResources.getDirectory(ResourceType.DATASET, localCacheName())
    }

    fun isCached(): Boolean {
        return getLocalCacheDir().exists()
    }


    private fun deleteIfEmpty(localCache: File) {
        if (localCache.exists()) {
            val files = localCache.listFiles()
            if (files == null || files.size < 1) {
                try {
                    FileUtils.deleteDirectory(localCache)
                } catch (e: IOException) {
                    logger.debug("Error deleting directory: {}", localCache)
                }
            }
        }
    }


    fun remoteDataUrl(set: DataSetType): String {
        return DL4JResources.getURLString("datasets/cifar10_dl4j.v1.zip")
    }

    fun localCacheName(): String {
        return LOCAL_CACHE_NAME
    }

    fun expectedChecksum(set: DataSetType): Long {
        return 292852033L
    }

    fun getRecordReader(
        rngSeed: Long,
        imgDim: IntArray?,
        set: CustomDataSetType,
        imageTransform: ImageTransform?,
        iteratorDistribution: List<Int>,
        maxSamples: Int,
    ): RecordReader {
        Preconditions.checkState(imgDim == null || imgDim.size == 2,
            "Invalid image dimensions: must be null or length 2. Got: %s",
            imgDim)
        // check empty cache
        val localCache = getLocalCacheDir()
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
        datasetPath = when (set) {
            CustomDataSetType.TRAIN -> File(localCache, "/train/")
            CustomDataSetType.TEST -> File(localCache, "/test/")
            CustomDataSetType.VALIDATION -> throw IllegalArgumentException("You will need to manually create and iterate a validation directory, CIFAR-10 does not provide labels")
            else -> File(localCache, "/train/")
        }

        // set up file paths
        val pathFilter = RandomPathFilter(rng, *BaseImageLoader.ALLOWED_FORMATS)
        val filesInDir = FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS, rng)
        // Randomly shuffled?
        val filesInDirSplit = filesInDir.sample(pathFilter, 1.0)
        val uris = HashMap<Int, MutableList<URI>>()
        for (label in iteratorDistribution.indices) {
            uris[label] = mutableListOf()
        }
        var count = 0
        val maxElements = iteratorDistribution.sum()
        for (uri in filesInDirSplit[0].locations()) {
            val split = uri.toString().split('/')
            val label = split[split.size - 2].toInt()
            val uriSet = uris[label]!!
            if (uriSet.size < min(maxSamples, iteratorDistribution[label])) {
                uriSet.add(uri)
                count++
                if (count >= maxElements) {
                    break
                }
            }
        }
        val newFilesInDirSplit = CollectionInputSplit(uris.values.toList().flatten().shuffled(rng))
        val h = imgDim?.get(0) ?: INPUT_HEIGHT
        val w = imgDim?.get(1) ?: INPUT_WIDTH
        val rr = CustomImageRecordReader(h.toLong(),
            w.toLong(),
            INPUT_CHANNELS.toLong(),
            ParentPathLabelGenerator(),
            imageTransform)
        try {
            rr.initialize(newFilesInDirSplit)
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
        return rr
    }

    companion object {
        const val LABELS_FILENAME = "labels.txt"
        const val LOCAL_CACHE_NAME = "cifar10"
        var INPUT_WIDTH = 32
        var INPUT_HEIGHT = 32
        var INPUT_CHANNELS = 3
        var NUM_LABELS = 10
    }
}
