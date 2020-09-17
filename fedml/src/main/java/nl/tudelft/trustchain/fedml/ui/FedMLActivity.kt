package nl.tudelft.trustchain.fedml.ui

import android.Manifest.permission
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
