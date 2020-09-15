package nl.tudelft.trustchain.fedml.ui

import android.content.pm.PackageManager
import android.os.Build
import android.widget.Toast
import androidx.core.app.ActivityCompat
import nl.tudelft.trustchain.common.BaseActivity
import nl.tudelft.trustchain.fedml.R

class FedMLActivity : BaseActivity() {
    override val navigationGraph = R.navigation.nav_graph_fedml

    override fun onStart() {
        super.onStart()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (ActivityCompat.checkSelfPermission(
                    applicationContext,
                    android.Manifest.permission.READ_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED || ActivityCompat.checkSelfPermission(
                    applicationContext,
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.READ_EXTERNAL_STORAGE, android.Manifest.permission.WRITE_EXTERNAL_STORAGE), PackageManager.PERMISSION_GRANTED)
                Toast.makeText(applicationContext, "error", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(applicationContext, "top", Toast.LENGTH_SHORT).show()
            }
        }
    }

    companion object {
        init {
//            System.loadLibrary("c++_shared")
        }
    }
}
