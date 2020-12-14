package nl.tudelft.trustchain.fedml.ai.dataset.cifar

import mu.KotlinLogging
import org.datavec.api.conf.Configuration
import org.datavec.api.io.labels.PathLabelGenerator
import org.datavec.api.io.labels.PathMultiLabelGenerator
import org.datavec.api.records.Record
import org.datavec.api.records.metadata.RecordMetaData
import org.datavec.api.records.metadata.RecordMetaDataURI
import org.datavec.api.records.reader.BaseRecordReader
import org.datavec.api.split.FileSplit
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

class CustomImageRecordReader(
    private var height: Long,
    private var width: Long,
    private var channels: Long,
    private var labelGenerator: PathLabelGenerator,
    private var imageTransform: ImageTransform?
) : BaseRecordReader() {
    private var finishedInputStreamSplit = false
    private var iter: Iterator<File>? = null
    private lateinit var conf: Configuration
    private var currentFile: File? = null
    private var labelMultiGenerator: PathMultiLabelGenerator? = null
    private var labels: MutableList<String> = arrayListOf()
    private var appendLabel = false
    private var writeLabel = false
    private var record: List<Writable>? = null
    private var hitImage = false
    private var cropImage = false
    private var imageLoader: BaseImageLoader? = null
    private var inputSplit: InputSplit? = null
    private var fileNameMap = LinkedHashMap<String, String>()
    private var pattern: String? = null
    private var patternPosition = 0

    private var logLabelCountOnInit = true

    private val HEIGHT = "$NAME_SPACE.height"
    private val WIDTH = "$NAME_SPACE.width"
    private val CHANNELS = "$NAME_SPACE.channels"
    private val CROP_IMAGE = "$NAME_SPACE.cropimage"
    private val IMAGE_LOADER = "$NAME_SPACE.imageloader"

    init {
        this.appendLabel = true
    }

    fun containsFormat(format: String): Boolean {
        for (format2 in imageLoader!!.allowedFormats) if (format.endsWith(".$format2")) return true
        return false
    }


    override fun initialize(split: InputSplit) {
        if (imageLoader == null) {
            imageLoader = NativeImageLoader(height, width, channels, imageTransform)
        }
        if (split is InputStreamInputSplit) {
            this.inputSplit = split
            this.finishedInputStreamSplit = false
            return
        }
        inputSplit = split
        val locations = split.locations()
        if (locations != null && locations.isNotEmpty()) {
            if (appendLabel && labelGenerator.inferLabelClasses()) {
                val labelsSet: MutableSet<String> = HashSet()
                for (location in locations) {
                    val imgFile = File(location)
                    val name = labelGenerator.getLabelForPath(location).toString()
                    labelsSet.add(name)
                    if (pattern != null) {
                        val label = name.split(pattern!!).toTypedArray()[patternPosition]
                        fileNameMap[imgFile.toString()] = label
                    }
                }
                labels.clear()
                labels.addAll(labelsSet)
                if (logLabelCountOnInit) {
                    logger.info("ImageRecordReader: ${labelsSet.size} label classes inferred using label generator ${labelGenerator.javaClass.simpleName}")
                }
            }
            iter = FileFromPathIterator(inputSplit.locationsPathIterator()) //This handles randomization internally if necessary
        } else throw IllegalArgumentException("No path locations found in the split.")
        if (split is FileSplit) {
            //remove the root directory
            labels.remove(split.rootDir)
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        labels.sort()
    }


    @Throws(IOException::class, InterruptedException::class)
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


    /**
     * Called once at initialization.
     *
     * @param split          the split that defines the range of records to read
     * @param imageTransform the image transform to use to transform images while loading them
     */
    fun initialize(split: InputSplit, imageTransform: ImageTransform?) {
        this.imageLoader = null
        this.imageTransform = imageTransform
        initialize(split)
    }

    /**
     * Called once at initialization.
     *
     * @param conf           a configuration for initialization
     * @param split          the split that defines the range of records to read
     * @param imageTransform the image transform to use to transform images while loading them
     * @throws java.io.IOException
     * @throws InterruptedException
     */
    fun initialize(conf: Configuration, split: InputSplit, imageTransform: ImageTransform?) {
        this.imageLoader = null
        this.imageTransform = imageTransform
        initialize(conf, split)
    }


    override fun next(): List<Writable>? {
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
                    if (labelMultiGenerator != null) {
                        ret.addAll(labelMultiGenerator!!.getLabels(image.path))
                    } else {
                        if (labelGenerator.inferLabelClasses()) {
                            //Standard classification use case (i.e., handle String -> integer conversion
                            ret.add(IntWritable(labels.indexOf(getLabel(image.path))))
                        } else {
                            //Regression use cases, and PathLabelGenerator instances that already map to integers
                            ret.add(labelGenerator.getLabelForPath(image.path))
                        }
                    }
                }
            } catch (e: Exception) {
                throw RuntimeException(e)
            }
            return ret
        } else if (record != null) {
            hitImage = true
            invokeListeners(record)
            return record
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

    override fun getLabels(): MutableList<String> {
        return labels
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
        var currLabelsWritable: MutableList<Writable>? = null
        var multiGenLabels: MutableList<List<Writable?>>? = null
        while (cnt < num && iter!!.hasNext()) {
            currentFile = iter!!.next()
            currBatch.add(currentFile!!)
            invokeListeners(currentFile)
            if (appendLabel || writeLabel) {
                //Collect the label Writables from the label generators
                if (labelMultiGenerator != null) {
                    if (multiGenLabels == null) multiGenLabels = ArrayList()
                    multiGenLabels.add(labelMultiGenerator!!.getLabels(currentFile!!.path))
                } else {
                    if (labelGenerator.inferLabelClasses()) {
                        if (currLabels == null) currLabels = ArrayList()
                        currLabels.add(labels.indexOf(getLabel(currentFile!!.path)))
                    } else {
                        if (currLabelsWritable == null) currLabelsWritable = ArrayList()
                        currLabelsWritable.add(labelGenerator.getLabelForPath(currentFile!!.path))
                    }
                }
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
            if (labelMultiGenerator != null) {
                val temp: MutableList<Writable?> = ArrayList()
                val first = multiGenLabels!![0]
                for (col in first.indices) {
                    temp.clear()
                    for (multiGenLabel in multiGenLabels) {
                        temp.add(multiGenLabel[col])
                    }
                    val currCol = RecordConverter.toMinibatchArray(temp)
                    ret.add(currCol)
                }
            } else {
                val labels: INDArray
                if (labelGenerator.inferLabelClasses()) {
                    //Standard classification use case (i.e., handle String -> integer conversion)
                    labels = Nd4j.create(cnt.toLong(), numCategories.toLong(), 'c')
                    Nd4j.getAffinityManager().tagLocation(labels, AffinityManager.Location.HOST)
                    for (i in currLabels!!.indices) {
                        labels.putScalar(i.toLong(), currLabels[i]!!.toLong(), 1.0)
                    }
                } else {
                    //Regression use cases, and PathLabelGenerator instances that already map to integers
                    labels = if (currLabelsWritable!![0] is NDArrayWritable) {
                        val arr: MutableList<INDArray> = ArrayList()
                        for (w in currLabelsWritable) {
                            arr.add((w as NDArrayWritable).get())
                        }
                        Nd4j.concat(0, *arr.toTypedArray())
                    } else {
                        RecordConverter.toMinibatchArray(currLabelsWritable)
                    }
                }
                ret.add(labels)
            }
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
    fun getLabel(path: String?): String {
        return labelGenerator.getLabelForPath(path).toString()
    }

    /**
     * Accumulate the label from the path
     *
     * @param path the path to get the label from
     */
    fun accumulateLabel(path: String?) {
        val name = getLabel(path)
        if (!labels.contains(name)) labels.add(name)
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

    fun numLabels(): Int {
        return labels.size
    }

    @Throws(IOException::class)
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

    @Throws(IOException::class)
    override fun loadFromMetaData(recordMetaData: RecordMetaData): Record {
        return loadFromMetaData(listOf(recordMetaData))[0]
    }

    @Throws(IOException::class)
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
