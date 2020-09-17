package nl.tudelft.trustchain.fedml.ui

import android.os.Bundle
import android.os.StrictMode
import android.view.View
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.ai.MNISTLocalRunner
import nl.tudelft.trustchain.fedml.ai.MNISTSimulatedRunner
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
        return getIpv8().getOverlay() ?: throw java.lang.IllegalStateException("FedMLCommunity is not configured")
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        binding.btnPing.setOnClickListener { onBtnPingClicked() }
        binding.btnMnistLocal.setOnClickListener { onBtnRunMNISTLocallyClicked() }
        binding.btnMnistDistrSim.setOnClickListener { onBtnSimulateDistributedMNISTLocallyClicked() }
        binding.btnMnistDistr.setOnClickListener { onBtnRunMNISTDistributedClicked() }

        configureDL4JDirectory()
        allowDL4JOnUIThread()
    }

    private fun configureDL4JDirectory() {
        DL4JResources.setBaseDirectory(requireActivity().getExternalFilesDir(null)!!)
    }

    private fun allowDL4JOnUIThread() {
        val policy: StrictMode.ThreadPolicy = StrictMode.ThreadPolicy.Builder().permitAll().build()
        StrictMode.setThreadPolicy(policy)
    }

    ////// BUTTON CLICK LISTENERS

    private fun onBtnPingClicked() {
        getCommunity().sendToAll(FedMLCommunity.MessageId.MSG_PING, MsgPing("Ping"))
    }

    private fun onBtnRunMNISTLocallyClicked() {
        MNISTLocalRunner().run()
    }

    private fun onBtnSimulateDistributedMNISTLocallyClicked() {
        MNISTSimulatedRunner().run()
    }

    private fun onBtnRunMNISTDistributedClicked() {
        System.out.println("foo")
    }
}
