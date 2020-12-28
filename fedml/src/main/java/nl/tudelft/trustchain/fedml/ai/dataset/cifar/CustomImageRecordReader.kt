package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import org.datavec.api.conf.Configuration
import org.datavec.api.records.Record
import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.metadata.RecordMetaDataURI
import org.datavec.api.records.reader.BaseRecordReader
import org.datavec.api.split.InputSplit
import org.datavec.api.split.InputStreamInputSplit
import org.datavec.api.util.files.FileFromPathIterator
import org.datavec.api.util.files.URIUtil
import org.datavec.api.util.ndarray.RecordConverter
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.NDArrayWritable
import org.datavec.api.writable.Writable
import org.datavec.api.writable.batch.NDArrayRecordBatch
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.loader.ImageLoader
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.BaseImageRecordReader
import org.datavec.image.transform.ImageTransform
import org.nd4j.linalg.api.concurrency.AffinityManager
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.guava.base.Preconditions
import java.io.*
import java.net.URI
import java.util.*

private val logger = KotlinLogging.logger("CustomImageRecordReader")

/**
 * Specific for CIFAR-10
 */
class CustomImageRecordReader(
    private var height: Long,
    private var width: Long,
    private var channels: Long,
    private var imageTransform: ImageTransform?,
) : BaseRecordReader() {
    private lateinit var conf: Configuration
    private var finishedInputStreamSplit = false
    private var iter: Iterator<File>? = null
    private var currentFile: File? = null
    private var labels = listOf("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    private var uniqueLabels = mutableSetOf<String>()
    private var appendLabel = true
    private var writeLabel = false
    private var record: List<Writable>? = null
    private var hitImage = false
    private var cropImage = false
    private var imageLoader: BaseImageLoader? = null
    private val HEIGHT = "$NAME_SPACE.height"
    private val WIDTH = "$NAME_SPACE.width"
    private val CHANNELS = "$NAME_SPACE.channels"
    private val CROP_IMAGE = "$NAME_SPACE.cropimage"
    private val IMAGE_LOADER = "$NAME_SPACE.imageloader"

    override fun initialize(split: InputSplit) {
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        inputSplit = split
        iter = FileFromPathIterator(inputSplit.locationsPathIterator()) //This handles randomization internally if necessary
        val locations = split.locations()
        for (location in locations) {
            val a = location.toString().split('/')
            val label = a[a.size - 2]
            uniqueLabels.add(label)
        }
    }


    override fun initialize(conf: Configuration, split: InputSplit) {
        this.appendLabel = conf.getBoolean(APPEND_LABEL, appendLabel)
        this.labels = ArrayList(conf.getStringCollection(LABELS))
        this.height = conf.getLong(HEIGHT, height)
        width = conf.getLong(WIDTH, width)
        channels = conf.getLong(CHANNELS, channels)
        this.cropImage = conf.getBoolean(CROP_IMAGE, cropImage)
        if ("imageio" == conf[IMAGE_LOADER]) {
            this.imageLoader = ImageLoader(height, width, channels, cropImage)
        } else {
            this.imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        this.conf = conf
        initialize(split)
    }

    override fun next(): List<Writable> {
        if (inputSplit is InputStreamInputSplit) {
            val inputStreamInputSplit = inputSplit as InputStreamInputSplit
            try {
                val ndArrayWritable = NDArrayWritable(imageLoader!!.asMatrix(inputStreamInputSplit.getIs()))
                finishedInputStreamSplit = true
                return listOf<Writable>(ndArrayWritable)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
        if (iter != null) {
            val ret: MutableList<Writable>
            val image = iter!!.next()
            currentFile = image
            if (image.isDirectory) return next()
            try {
                invokeListeners(image)
                val row = imageLoader!!.asMatrix(image)
                Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE)
                ret = RecordConverter.toRecord(row)
                if (appendLabel || writeLabel) {
                    ret.add(IntWritable(labels.indexOf(getLabel(image.path))))
                }
            } catch (e: Exception) {
                e.printStackTrace()
                throw RuntimeException(e)
            }
            return ret
        } else if (record != null) {
            hitImage = true
            invokeListeners(record)
            return record!!
        }
        throw IllegalStateException("No more elements")
    }

    override fun hasNext(): Boolean {
        if (inputSplit is InputStreamInputSplit) {
            return finishedInputStreamSplit
        }
        if (iter != null) {
            return iter!!.hasNext()
        } else if (record != null) {
            return !hitImage
        }
        throw IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist")
    }

    override fun getLabels(): List<String> {
        return labels
    }

    fun getUniqueLabels(): Set<String> {
        return uniqueLabels
    }

    override fun batchesSupported(): Boolean {
        return imageLoader is NativeImageLoader
    }

    override fun next(num: Int): List<List<Writable?>?> {
        Preconditions.checkArgument(num > 0, "Number of examples must be > 0: got $num")
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        val currBatch: MutableList<File> = ArrayList()
        var cnt = 0
        val numCategories = if (appendLabel || writeLabel) labels.size else 0
        var currLabels: MutableList<Int?>? = null
        while (cnt < num && iter!!.hasNext()) {
            currentFile = iter!!.next()
            currBatch.add(currentFile!!)
            invokeListeners(currentFile)
            if (appendLabel || writeLabel) {
                //Collect the label Writables from the label generators
                if (currLabels == null) currLabels = ArrayList()
                currLabels.add(labels.indexOf(getLabel(currentFile!!.path)))
            }
            cnt++
        }
        val features = Nd4j.createUninitialized(longArrayOf(cnt.toLong(), channels, height, width), 'c')
        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST)
        for (i in 0 until cnt) {
            try {
                (imageLoader as NativeImageLoader).asMatrixView(currBatch[i],
                    features.tensorAlongDimension(i.toLong(), 1, 2, 3))
            } catch (e: Exception) {
                println("Image file failed during load: " + currBatch[i].absolutePath)
                throw RuntimeException(e)
            }
        }
        Nd4j.getAffinityManager().ensureLocation(features, AffinityManager.Location.DEVICE)
        val ret: MutableList<INDArray> = ArrayList()
        ret.add(features)
        if (appendLabel || writeLabel) {
            //And convert the previously collected label Writables from the label generators
            //Standard classification use case (i.e., handle String -> integer conversion)
            val labels = Nd4j.create(cnt.toLong(), numCategories.toLong(), 'c')
            Nd4j.getAffinityManager().tagLocation(labels, AffinityManager.Location.HOST)
            for (i in currLabels!!.indices) {
                labels.putScalar(i.toLong(), currLabels[i]!!.toLong(), 1.0)
            }
            ret.add(labels)
        }
        return NDArrayRecordBatch(ret)
    }

    @Throws(IOException::class)
    override fun close() {
        //No op
    }

    override fun setConf(conf: Configuration) {
        this.conf = conf
    }

    override fun getConf(): Configuration {
        return conf
    }

    /**
     * Get the label from the given path
     *
     * @param path the path to get the label from
     * @return the label for the given path
     */
    private fun getLabel(path: String): String {
        val split = path.split('/')
        return split[split.size - 2]
    }

    override fun reset() {
        if (inputSplit == null) throw UnsupportedOperationException("Cannot reset without first initializing")
        inputSplit.reset()
        if (iter != null) {
            iter = FileFromPathIterator(inputSplit.locationsPathIterator())
        } else if (record != null) {
            hitImage = false
        }
    }

    override fun resetSupported(): Boolean {
        return if (inputSplit == null) {
            false
        } else inputSplit.resetSupported()
    }

    override fun record(uri: URI, dataInputStream: DataInputStream?): List<Writable>? {
        invokeListeners(uri)
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        val row = imageLoader!!.asMatrix(dataInputStream)
        val ret = RecordConverter.toRecord(row)
        if (appendLabel) ret.add(IntWritable(labels.indexOf(getLabel(uri.path))))
        return ret
    }

    override fun nextRecord(): Record {
        val list = next()
        val uri = URIUtil.fileToURI(currentFile)
        return org.datavec.api.records.impl.Record(list, RecordMetaDataURI(uri, BaseImageRecordReader::class.java))
    }

    override fun loadFromMetaData(recordMetaData: RecordMetaData): Record {
        return loadFromMetaData(listOf(recordMetaData))[0]
    }

    override fun loadFromMetaData(recordMetaDatas: List<RecordMetaData>): List<Record> {
        val out: MutableList<Record> = ArrayList()
        for (meta in recordMetaDatas) {
            val uri = meta.uri
            val f = File(uri)
            var next: List<Writable>?
            DataInputStream(BufferedInputStream(FileInputStream(f))).use { dis -> next = record(uri, dis) }
            out.add(org.datavec.api.records.impl.Record(next, meta))
        }
        return out
    }
}
