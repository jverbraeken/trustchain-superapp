package nl.tudelft.trustchain.app

import android.app.Activity
import androidx.annotation.ColorRes
import androidx.annotation.DrawableRes
import com.example.musicdao.MusicService
import nl.tudelft.trustchain.FOC.MainActivityFOC
import nl.tudelft.trustchain.common.R
import nl.tudelft.trustchain.currencyii.CurrencyIIMainActivity
import nl.tudelft.trustchain.debug.DebugActivity
import nl.tudelft.trustchain.distributedAI.DistributedActivity
import nl.tudelft.trustchain.explorer.ui.TrustChainExplorerActivity
import nl.tudelft.trustchain.trader.ui.TrustChainTraderActivity
import nl.tudelft.trustchain.payloadgenerator.ui.TrustChainPayloadGeneratorActivity
import nl.tudelft.trustchain.peerchat.PeerChatActivity
import nl.tudelft.trustchain.voting.VotingActivity
import nl.tudelft.trustchain.fedml.ui.FedMLActivity

enum class AppDefinition(
    val id: String,
    @DrawableRes val icon: Int,
    val appName: String,
    @ColorRes val color: Int,
    val activity: Class<out Activity>
) {
    PEERCHAT(
        "peerchat",
        R.drawable.ic_chat_black_24dp,
        "PeerChat",
        R.color.purple,
        PeerChatActivity::class.java
    ),
    TRUSTCHAIN_EXPLORER(
        "explorer",
        R.drawable.ic_device_hub_black_24dp,
        "TrustChain Explorer",
        R.color.red,
        TrustChainExplorerActivity::class.java
    ),
    DEBUG(
        "debug",
        R.drawable.ic_bug_report_black_24dp,
        "Debug",
        R.color.dark_gray,
        DebugActivity::class.java
    ),
    CURRENCY_II(
        "currencyii",
        R.drawable.ic_bitcoin,
        "Luxury Communism",
        R.color.metallic_gold,
        CurrencyIIMainActivity::class.java
    ),
    TRUSTCHAIN_TRADER(
        "trustchaintrader",
        R.drawable.ic_device_hub_black_24dp,
        "AI trading bot",
        R.color.blue,
        TrustChainTraderActivity::class.java
    ),
    TRUSTCHAIN_PAYLOADGENERATOR(
        "payloadgenerator",
        R.drawable.ic_add_black_24dp,
        "Market Bot",
        R.color.black,
        TrustChainPayloadGeneratorActivity::class.java
    ),
    FREEDOM_OF_COMPUTING(
        "foc",
        R.drawable.ic_naruto,
        "Freedom of Computing",
        R.color.blue,
        MainActivityFOC::class.java
    ),
    DNA(
        "dna",
        R.drawable.ic_bug_report_black_24dp,
        "Distributed AI",
        R.color.red,
        DistributedActivity::class.java
    ),
    VOTING(
        "voting",
        R.drawable.abc_ic_voice_search_api_material,
        "Voter",
        R.color.android_green,
        VotingActivity::class.java
    ),
    MUSIC_DAO(
        "musicdao",
        android.R.drawable.ic_media_play,
        "MusicDAO",
        R.color.black,
        MusicService::class.java
    ),
    FEDML(
        "fedml",
        R.drawable.ic_device_hub_black_24dp,
        "Federated ML Platform",
        R.color.red,
        FedMLActivity::class.java
    )
}
