package nl.tudelft.trustchain.fedml.ui

import android.os.Bundle
import android.view.View
import nl.tudelft.trustchain.common.ui.BaseFragment
import nl.tudelft.trustchain.common.util.viewBinding
import nl.tudelft.trustchain.fedml.R
import nl.tudelft.trustchain.fedml.databinding.FragmentTestBinding
import nl.tudelft.trustchain.fedml.ipv8.FedMLCommunity
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
            val tmp = CifarDataSetService()
            tmp.c()
        }

//        val network = CnnModel(CifarDataSetService(), CnnModelProperties())
//
//        network.train()
//        val evaluation: Evaluation = network.evaluate()
//        print(evaluation)
    }
}
