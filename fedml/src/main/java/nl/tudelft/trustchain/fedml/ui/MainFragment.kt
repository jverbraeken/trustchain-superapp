package nl.tudelft.trustchain.fedml.ui

import android.os.Bundle
import android.os.StrictMode
import android.view.View
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.ai.MNISTDistributedRunner
import nl.tudelft.trustchain.fedml.ai.MNISTLocalRunner
import nl.tudelft.trustchain.fedml.ai.MNISTSimulatedRunner
import nl.tudelft.trustchain.fedml.databinding.FragmentMainBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity.MessageId
import nl.tudelft.trustchain.fedml.ipv8.MsgPing
import org.deeplearning4j.common.resources.DL4JResources

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
        getCommunity().sendToAll(MessageId.MSG_PING, MsgPing("Ping"))
    }

    private fun onBtnRunMNISTLocallyClicked() {
        MNISTLocalRunner().run()
    }

    private fun onBtnSimulateDistributedMNISTLocallyClicked() {
        MNISTSimulatedRunner().run()
    }

    private fun onBtnRunMNISTDistributedClicked() {
        MNISTDistributedRunner(getCommunity()).run()
    }
}
