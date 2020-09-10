package nl.tudelft.trustchain.fedml.ipv8

import android.util.Log
import nl.tudelft.ipv8.IPv4Address
import nl.tudelft.ipv8.Overlay
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCommunity
import nl.tudelft.ipv8.attestation.trustchain.TrustChainCrawler
import nl.tudelft.ipv8.attestation.trustchain.TrustChainSettings
import nl.tudelft.ipv8.attestation.trustchain.store.TrustChainStore
import nl.tudelft.ipv8.messaging.Deserializable
import nl.tudelft.ipv8.messaging.Packet

class FedMLCommunity(
    settings: TrustChainSettings,
    database: TrustChainStore,
    crawler: TrustChainCrawler = TrustChainCrawler()
) : TrustChainCommunity(settings, database, crawler) {
    override val serviceId = "29384902d2938f34872398758cf7ca9238ccc333"

    class Factory(
        private val settings: TrustChainSettings,
        private val database: TrustChainStore,
        private val crawler: TrustChainCrawler = TrustChainCrawler()
    ) : Overlay.Factory<FedMLCommunity>(
        FedMLCommunity::class.java
    ) {
        override fun create(): FedMLCommunity {
            return FedMLCommunity(
                settings,
                database,
                crawler
            )
        }
    }

    override fun bootstrap() {
        super.bootstrap()

        for (address in INITIAL_ADDRESSES) {
            walkTo(address)
        }
    }

    object MessageId {
        const val THALIS_MESSAGE = 222
        const val TORRENT_MESSAGE = 223
        const val PUNCTURE_TEST = 251
    }

    fun sendMessage() {
        for (peer in getPeers()) {
            val packet = serializePacket(
                MessageId.THALIS_MESSAGE,
                MyMessage("Hello from Freedom of Computing!"),
                true
            )
            send(peer.address, packet)
        }
    }

    // RECEIVE MESSAGE
    init {
        messageHandlers[MessageId.THALIS_MESSAGE] = ::onMessage
    }

    private fun onMessage(packet: Packet) {
        val (peer, payload) = packet.getAuthPayload(MyMessage.Deserializer)
        Log.e("personal", peer.mid + ": " + payload.message)
    }

    companion object {
        // These are initial addresses for some peers that have initial content,
        // in the case that no content can be found on the first run of the app.
        val INITIAL_ADDRESSES: List<IPv4Address> = listOf(
            IPv4Address("83.84.32.175", 35376)
        )
    }
}


data class MyMessage(val message: String) : nl.tudelft.ipv8.messaging.Serializable {
    override fun serialize(): ByteArray {
        return message.toByteArray()
    }

    companion object Deserializer : Deserializable<MyMessage> {
        override fun deserialize(buffer: ByteArray, offset: Int): Pair<MyMessage, Int> {
            val toReturn = buffer.toString(Charsets.UTF_8)
            return Pair(MyMessage(toReturn), buffer.size)
        }
    }
}
