package nl.tudelft.trustchain.fedml.ui

import android.os.Bundle
import android.os.StrictMode
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.databinding.FragmentMainBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.MsgPing
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
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

class MainFragment : BaseFragment(R.layout.fragment_main) {
    private val binding by viewBinding(FragmentMainBinding::bind)

    private fun getCommunity(): FedMLCommunity {
        return getIpv8().getOverlay()
            ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.btnPing.setOnClickListener { onBtnPingClicked() }
        binding.btnAI.setOnClickListener { onBtnAIClicked() }
    }

    private fun onBtnPingClicked() {
        getCommunity().sendToAll(FedMLCommunity.MessageId.MSG_PING, MsgPing("Ping"))
    }

    private fun onBtnAIClicked() {
        val nChannels = 1
        val outputNum = 10
        val batchSize = 64
        val seed = 123L

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

        val networks = arrayOf(MultiLayerNetwork(conf), MultiLayerNetwork(conf))
        networks.forEach {
            it.init()
            it.setListeners(ScoreIterationListener(1))
        }
        for (i in 0..100) {
            for (net in networks) {
                net.fit(mnistTrain.next())
            }
            var arr: INDArray? = null
            for (j in networks.indices) {
                if (j == 0) {
                    arr = networks[j].params()
                } else {
                    arr = arr!!.add(networks[j].params())
                }
            }
            arr = arr!!.divi(networks.size)
            for (net in networks) {
                net.setParameters(arr!!)
            }
        }
    }
}
