package nl.tudelft.trustchain.fedml.ui

import android.content.res.AssetManager
import android.os.Bundle
import android.os.StrictMode
import android.util.Log
import android.view.View
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.databinding.FragmentTestBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import org.deeplearning4j.common.resources.DL4JResources
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.*
import java.util.*

class TestFragment : BaseFragment(R.layout.fragment_test) {
    private val binding by viewBinding(FragmentTestBinding::bind)

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.btnInitiate.setOnClickListener {
//            getCommunity().sendMessage()
//            copyAssets()

//            val randNumGen = Random(1234)
            /*val trainData = File(requireActivity().getExternalFilesDir(null), "training")
            val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
            val labelMaker = ParentPathLabelGenerator()
            val trainRR = ImageRecordReader(28, 28, 1, labelMaker)
            trainRR.initialize(trainSplit)
            val trainIter = RecordReaderDataSetIterator(trainRR, 64, 1, 10)

            val scaler = ImagePreProcessingScaler()
            scaler.fit(trainIter)
            trainIter.preProcessor = scaler

            val testData = File(requireActivity().getExternalFilesDir(null), "testing")
            val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
            val labelMaker2 = ParentPathLabelGenerator()
            val testRR = ImageRecordReader(28, 28, 1, labelMaker2)
            testRR.initialize(testSplit)
            val testIter = RecordReaderDataSetIterator(testRR, 64, 1, 10)
            testIter.preProcessor = scaler*/

            val nChannels = 1 // Number of input channels

            val outputNum = 10 // The number of possible outcomes

            val batchSize = 64 // Test batch size

            val nEpochs = 1 // Number of training epochs

            val seed = 123L //


            DL4JResources.setBaseDirectory(requireActivity().getExternalFilesDir(null)!!)
            val policy: StrictMode.ThreadPolicy = StrictMode.ThreadPolicy.Builder().permitAll().build()
            StrictMode.setThreadPolicy(policy)

            val mnistTrain = MnistDataSetIterator(batchSize, true, 12345)
//            val mnistTest = MnistDataSetIterator(batchSize, false, 12345)

            /*val learningRateSchedule: MutableMap<Int, Double> = HashMap()
            learningRateSchedule.put(0, 0.06)
            learningRateSchedule.put(200, 0.05)
            learningRateSchedule.put(600, 0.028)
            learningRateSchedule.put(800, 0.0060)
            learningRateSchedule.put(1000, 0.001)*/

            val conf = NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam(1e-3)/*Nesterovs(MapSchedule(ScheduleType.ITERATION, learningRateSchedule))*/)
                .list()
                .layer(
                    ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build()
                )
                .layer(
                    SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                .layer(
                    ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build()
                )
                .layer(
                    SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                .layer(
                    DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(500)
                        .build()
                )
                .layer(
                    OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build()
                )
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build()

//            val uiServer = UIServer.getInstance()
//            val statsStorage = InMemoryStatsStorage()
//            uiServer.attach(statsStorage)

            val network = MultiLayerNetwork(conf)
            network.init()
            network.setListeners(ScoreIterationListener(5)/*, EvaluativeListener(testIter, 1)*//*, StatsListener(statsStorage)*/)
            network.fit(mnistTrain, nEpochs)
//            val eval = Evaluation(10)
//            while (testIter.hasNext()) {
//                val next = testIter.next()
//                val output = network.output(next.features)
//                eval.eval(next.labels, output)
//            }
//            val result = network.evaluate<Evaluation>(testIter)
//            print(eval.stats())
//            print(result.axis)
//            network.save(File(requireActivity().getExternalFilesDir(null), "mnist-model.bin"))
        }

//        val network = CnnModel(CifarDataSetService(), CnnModelProperties())
//
//        network.train()
//        val evaluation: Evaluation = network.evaluate()
//        print(evaluation)
    }

    private fun copyAssets() {
        val assetManager: AssetManager = requireActivity().assets
        for (type in arrayOf("training", "testing")) {
            for (num in 0..9) {
                val path = "$type/$num/"
                var files: Array<String>? = null
                try {
                    files = assetManager.list(path)
                } catch (e: IOException) {
                    Log.e("tag", "Failed to get asset file list.", e)
                }
                val dir = File(requireActivity().getExternalFilesDir(null), path)
                if (!dir.exists()) {
                    dir.mkdirs()
                }
                if (files != null) {
                    for (filename in files) {
                        assetManager.open(path + filename).use { input ->
                            FileOutputStream(File(dir, filename)).use { output ->
                                copyFile(input, output)
                            }
                        }
                    }
                }
            }
        }
    }

    private fun copyFile(inn: InputStream, out: OutputStream) {
        out.use { fileOut -> inn.copyTo(fileOut) }
    }
}
