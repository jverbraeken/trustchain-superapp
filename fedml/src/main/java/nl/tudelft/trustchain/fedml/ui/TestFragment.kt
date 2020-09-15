package nl.tudelft.trustchain.fedml.ui

import android.content.res.AssetManager
import android.os.Bundle
import android.util.Log
import android.view.View
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.databinding.FragmentTestBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
//import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.*
import java.util.*
import kotlin.collections.HashMap


//import org.deeplearning4j.eval.Evaluation

class TestFragment : BaseFragment(R.layout.fragment_test) {
    private val binding by viewBinding(FragmentTestBinding::bind)

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.btnInitiate.setOnClickListener {
            getCommunity().sendMessage()
            copyAssets()

            val randNumGen = Random(1234)
            val trainData = File(requireActivity().getExternalFilesDir(null), "training")
            val trainSplit = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
            val labelMaker = ParentPathLabelGenerator()
            val trainRR = ImageRecordReader(28, 28, 1, labelMaker)
            trainRR.initialize(trainSplit)
            val trainIter = RecordReaderDataSetIterator(trainRR, 54, 1, 10)

            val scaler = ImagePreProcessingScaler()
            scaler.fit(trainIter)
            trainIter.preProcessor = scaler

            val testData = File(requireActivity().getExternalFilesDir(null), "testing")
            val testSplit = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen)
            val testRR = ImageRecordReader(28, 28, 1, labelMaker)
            testRR.initialize(testSplit)
            val testIter = RecordReaderDataSetIterator(testRR, 54, 1, 10)
            testIter.preProcessor = scaler

//            val mnistTrain = MnistDataSetIterator(54, true, 1234)
//            val mnistTest = MnistDataSetIterator(54, false, 1234)

            val lrSchedule = HashMap<Int, Double>()
            lrSchedule.put(0, 0.06)
            lrSchedule.put(200, 0.05)
            lrSchedule.put(600, 0.028)
            lrSchedule.put(800, 0.006)
            lrSchedule.put(1000, 0.001)

            val conf = NeuralNetConfiguration.Builder()
                .seed(1234)
                .l2(1e-4)
                .updater(Nesterovs(0.006, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,
                    ConvolutionLayer.Builder(5, 5)
                    .nIn(1)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(1,
                    SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(2, ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1)
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .build())
                .layer(3, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
                .layer(4, DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nOut(500)
                    .build())
                .layer(5, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(10)
                    .activation(Activation.SOFTMAX)
                    .build())
                /*.layer(
                    0, DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(
                    1, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(1000)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )*/
                .setInputType(InputType.convolutionalFlat(28, 28, 1))
                .build()

            val network = MultiLayerNetwork(conf)
            network.init()
//            network.setListeners(ScoreIterationListener(10), EvaluativeListener(testIter, 1, InvocationType.EPOCH_END))
            network.fit(trainIter)
//            val eval = Evaluation(10)
//            while (testIter.hasNext()) {
//                val next = testIter.next()
//                val output = network.output(next.features)
//                eval.eval(next.labels, output)
//            }
//            network.evaluate(mnistTest)
//            print(eval.stats())
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
        for (type in arrayOf("training", "test")) {
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
                if (files != null) for (filename in files) {
                    assetManager.open(path + filename).use { input ->
                        FileOutputStream(File(dir, filename)).use { output ->
                            copyFile(input, output)
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
