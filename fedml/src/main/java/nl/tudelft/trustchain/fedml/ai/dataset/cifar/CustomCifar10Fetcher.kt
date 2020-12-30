package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import nl.tudelft.trustchain.fedml.ai.CustomDataSetType
import org.apache.commons.io.FileUtils
import org.datavec.api.io.filters.RandomPathFilter
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.split.CollectionInputSplit
import org.datavec.api.split.FileSplit
import org.datavec.api.split.InputSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.transform.ImageTransform
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.common.resources.ResourceType
import org.nd4j.common.base.Preconditions
import org.nd4j.common.util.ArchiveUtils
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
    private fun downloadAndExtract() {
        val localFilename = "cifar10_dl4j.v1.zip"
        val tmpFile = File(System.getProperty("java.io.tmpdir"), localFilename)
        val localCacheDir = getLocalCacheDir()

        // Check empty cache
        if (localCacheDir.exists()) {
            val list = localCacheDir.listFiles()
            if (list == null || list.isEmpty()) localCacheDir.delete()
        }
        if (!localCacheDir.exists()) {
            localCacheDir.mkdirs()
            tmpFile.delete()
            logger.info("Downloading dataset to " + tmpFile.absolutePath)
            FileUtils.copyURLToFile(URL(REMOTE_DATA_URL), tmpFile)
        } else {
            // Directory exists and is non-empty - assume OK
            logger.info("Using cached dataset at " + localCacheDir.absolutePath)
            return
        }
        logger.info("Verifying download...")
        val adler: Checksum = Adler32()
        FileUtils.checksum(tmpFile, adler)
        val localChecksum = adler.value
        logger.info("Checksum local is $localChecksum, expecting $EXPECTED_CHECKSUM")
        if (localChecksum != EXPECTED_CHECKSUM) {
            logger.error("Checksums do not match. Cleaning up files and failing...")
            tmpFile.delete()
            throw IllegalStateException("Dataset file failed checksum: $tmpFile - expected checksum $EXPECTED_CHECKSUM vs. actual checksum $localChecksum. If this error persists, please open an issue at https://github.com/deeplearning4j/deeplearning4j.")
        }
        try {
            ArchiveUtils.unzipFileTo(tmpFile.absolutePath, localCacheDir.absolutePath, false)
        } catch (t: Throwable) {
            // Catch any errors during extraction, and delete the directory to avoid leaving the dir in an invalid state
            if (localCacheDir.exists()) FileUtils.deleteDirectory(localCacheDir)
            throw t
        }
    }

    private fun getLocalCacheDir(): File {
        return DL4JResources.getDirectory(ResourceType.DATASET, LOCAL_CACHE_NAME)
    }

    private fun deleteIfEmpty(localCache: File) {
        if (localCache.exists()) {
            val files = localCache.listFiles()
            if (files == null || files.isEmpty()) {
                try {
                    FileUtils.deleteDirectory(localCache)
                } catch (e: IOException) {
                    logger.debug("Error deleting directory: {}", localCache)
                }
            }
        }
    }

    @Synchronized
    fun getRecordReader(
        rngSeed: Long,
        imgDim: IntArray?,
        set: CustomDataSetType,
        imageTransform: ImageTransform?,
        iteratorDistribution: IntArray,
        maxSamples: Int,
    ): RecordReader {
        Preconditions.checkState(
            imgDim == null || imgDim.size == 2,
            "Invalid image dimensions: must be null or length 2. Got: %s",
            imgDim
        )
        // check empty cache
        val localCache = getLocalCacheDir()
        deleteIfEmpty(localCache)
        try {
            if (!localCache.exists()) {
                logger.debug { "Creating cache..." }
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
        if (pathFilter == null) {
            pathFilter = RandomPathFilter(rng, *BaseImageLoader.ALLOWED_FORMATS)
            filesInDir = FileSplit(datasetPath, BaseImageLoader.ALLOWED_FORMATS)
            filesInDirSplit = filesInDir!!.sample(pathFilter, 1.0)
        }
        val uris = HashMap<Int, Array<URI>>()
        var totalCount = 0
        val countPerLabel = IntArray(10)
        val maxTotalElements = iteratorDistribution.sum()
        val maxElementsPerLabel = iteratorDistribution.map { min(maxSamples, it) }
        val placeholderUri = URI("")
        for (label in iteratorDistribution.indices) {
            uris[label] = Array(maxElementsPerLabel[label]) { placeholderUri }
        }
        val locations = filesInDirSplit!![0].locations()
        val random = kotlin.random.Random(rngSeed)
        for (uri in locations) {
            val split = uri.toString().split('/')
            val label = split[split.size - 2].toInt()
            if (countPerLabel[label] < maxElementsPerLabel[label]) {
                uris[label]!![countPerLabel[label]] = uri
                countPerLabel[label]++
                totalCount++
                if (totalCount >= maxTotalElements) {
                    break
                }
            }
        }
        val newFilesInDirSplit = CollectionInputSplit(uris.values.toList().flatMap { it.asIterable() }.shuffled(random))

        val h = imgDim?.get(0) ?: INPUT_HEIGHT
        val w = imgDim?.get(1) ?: INPUT_WIDTH
        val rr = CustomImageRecordReader(
            h.toLong(),
            w.toLong(),
            INPUT_CHANNELS.toLong(),
            imageTransform
        )
        try {
            rr.initialize(newFilesInDirSplit)
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
        return rr
    }

    companion object {
        private const val LOCAL_CACHE_NAME = "cifar10"
        private const val EXPECTED_CHECKSUM = 292852033L
        private const val REMOTE_DATA_URL = "https://api.onedrive.com/v1.0/shares/s!AvNMRY4ml2WPgZpmYiFDOhNyFgakRQ/root/content"
        const val INPUT_WIDTH = 32
        const val INPUT_HEIGHT = 32
        const val INPUT_CHANNELS = 3
        const val NUM_LABELS = 10

        private var pathFilter: RandomPathFilter? = null
        private var filesInDir: FileSplit? = null
        private var filesInDirSplit: Array<out InputSplit>? = null
    }
}
