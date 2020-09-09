package nl.tudelft.trustchain.fedml.ui.blocks

class MyChainFragment : BlocksFragment() {
    override val isCrawlAllowed = false

    override fun getPublicKey(): ByteArray {
        return getTrustChainCommunity().myPeer.publicKey.keyToBin()
    }
}
