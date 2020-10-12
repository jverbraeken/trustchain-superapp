package nl.tudelft.trustchain.fedml.ui

import android.Manifest.permission
import android.R.attr.fragment
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.core.app.ActivityCompat
import nl.tudelft.trustchain.common.BaseActivity
import nl.tudelft.trustchain.fedml.R


class FedMLActivity : BaseActivity() {
    override val navigationGraph = R.navigation.nav_graph_fedml

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dataset = intent.extras?.getString("dataset")
//        val optimizer = intent.extras?.getString("optimizer")
//        val learningRate = intent.extras?.getString("learningRate")
//        val momentum = intent.extras?.getString("momentum")
//        val l2Regularization = intent.extras?.getString("l2Regularization")
//        val batchSize = intent.extras?.getString("batchSize")
//        val epoch = intent.extras?.getString("epoch")
        //        val runner = intent.extras?.getString("runner")

        val bundle = Bundle()
        bundle.putString("dataset", dataset)
        (MainFragment()).arguments = bundle
    }

    override fun onStart() {
        super.onStart()

        if (needToAskPermissions()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(permission.READ_EXTERNAL_STORAGE, permission.WRITE_EXTERNAL_STORAGE),
                PackageManager.PERMISSION_GRANTED
            )
            Toast.makeText(applicationContext, "Permissions requested", Toast.LENGTH_SHORT).show()
        }
    }

    private fun needToAskPermissions(): Boolean {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && (
            ActivityCompat.checkSelfPermission(applicationContext, permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED ||
                ActivityCompat.checkSelfPermission(applicationContext, permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)
    }
}
