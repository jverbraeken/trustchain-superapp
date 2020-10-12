package nl.tudelft.trustchain.app.ui.dashboard

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.GridLayoutManager
import com.mattskala.itemadapter.ItemAdapter
import mu.KotlinLogging
import nl.tudelft.trustchain.app.AppDefinition
import nl.tudelft.trustchain.app.databinding.ActivityDashboardBinding
import nl.tudelft.trustchain.common.util.viewBinding

private val logger = KotlinLogging.logger {}

class DashboardActivity : AppCompatActivity() {
    private val binding by viewBinding(ActivityDashboardBinding::inflate)

    private val adapter = ItemAdapter()

    init {
        adapter.registerRenderer(DashboardItemRenderer {
            val intent = Intent(this, it.app.activity)
            startActivity(intent)
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val launchActivity = intent.extras?.getString("activity")
        if (launchActivity != null) {
            logger.debug { "Activity flag" }
            startActivity(Intent(this, AppDefinition.values()
                .first { it.id == intent.extras?.getString("activity") }.activity).putExtras(intent.extras!!))
        }

        setContentView(binding.root)

        val layoutManager = GridLayoutManager(this, 3)
        binding.recyclerView.layoutManager = layoutManager
        binding.recyclerView.adapter = adapter

        adapter.updateItems(getAppList())
    }

    private fun getAppList(): List<DashboardItem> {
        return AppDefinition.values().map {
            DashboardItem(it)
        }
    }
}
