package nl.tudelft.ipv8.android.demo.coin

import android.util.Log
import com.google.common.base.Joiner
import info.blockchain.api.blockexplorer.BlockExplorer
import nl.tudelft.ipv8.util.hexToBytes
import nl.tudelft.ipv8.util.toHex
import org.bitcoinj.core.*
import org.bitcoinj.core.ECKey.ECDSASignature
import org.bitcoinj.core.listeners.DownloadProgressTracker
import org.bitcoinj.crypto.TransactionSignature
import org.bitcoinj.kits.WalletAppKit
import org.bitcoinj.params.MainNetParams
import org.bitcoinj.params.TestNet3Params
import org.bitcoinj.script.Script
import org.bitcoinj.script.ScriptBuilder
import org.bitcoinj.script.ScriptPattern
import org.bitcoinj.wallet.DeterministicSeed
import org.bitcoinj.wallet.SendRequest
import java.io.File
import java.util.*


/**
 * The wallet manager which encapsulates the functionality of all possible interactions
 * with bitcoin wallets (including multi-signature wallets).
 * NOTE: Ideally should be separated from any Android UI concepts. Not the case currently.
 */
class WalletManager(
    walletManagerConfiguration: WalletManagerConfiguration,
    walletDir: File,
    serializedDeterministicKey: SerializedDeterminsticKey? = null
) {
    val kit: WalletAppKit
    val params: NetworkParameters
    var isDownloading: Boolean = true
    var progress: Int = 0

    init {
        Log.i("Coin", "Coin: WalletManager attempting to start.")

        params = when (walletManagerConfiguration.network) {
            BitcoinNetworkOptions.TEST_NET -> TestNet3Params.get()
            BitcoinNetworkOptions.PRODUCTION -> MainNetParams.get()
        }

        val filePrefix = when (walletManagerConfiguration.network) {
            BitcoinNetworkOptions.TEST_NET -> "forwarding-service-testnet"
            BitcoinNetworkOptions.PRODUCTION -> "forwarding-service"
        }

        kit = object : WalletAppKit(params, walletDir, filePrefix) {
            override fun onSetupCompleted() {
                // Make a fresh new key if no keys in stored wallet.
                if (wallet().keyChainGroupSize < 1) wallet().importKey(ECKey())
                wallet().allowSpendingUnconfirmedTransactions()
                Log.i("Coin", "Coin: WalletManager started successfully.")
            }
        }

        if (serializedDeterministicKey != null) {
            Log.i(
                "Coin",
                "Coin: received a key to import, will clear the wallet and download again."
            )
            val deterministicSeed = DeterministicSeed(
                serializedDeterministicKey.seed,
                null,
                "",
                serializedDeterministicKey.creationTime
            )
            kit.restoreWalletFromSeed(deterministicSeed)
        }

        kit.setDownloadListener(object : DownloadProgressTracker() {
            override fun progress(
                pct: Double,
                blocksSoFar: Int,
                date: Date?
            ) {
                super.progress(pct, blocksSoFar, date)
                val percentage = pct.toInt()
                progress = percentage
                println("Progress: $percentage")
                Log.i("Coin", "Progress: $percentage")
            }

            override fun doneDownload() {
                super.doneDownload()
                Log.w("Coin", "Download Complete!")
                Log.i("Coin", "Balance: ${kit.wallet().balance}")
                isDownloading = false
            }
        })

        Log.i("Coin", "Coin: starting the setup of kit.")
        kit.setBlockingStartup(false)
        kit.startAsync()
        kit.awaitRunning()
        Log.i("Coin", "Coin: finished the setup of kit.")
    }

    /**
     * Returns our bitcoin public key we use in all multi-sig contracts
     * we are part of.
     * @return hex representation of our public key (this is not an address)
     */
    fun protocolECKey(): ECKey {
        return kit.wallet().issuedReceiveKeys[0]
    }

    /**
     * Returns our bitcoin public key (in hex) we use in all multi-sig contracts
     * we are part of.
     * @return hex representation of our public key (this is not an address)
     */
    fun networkPublicECKeyHex(): String {
        return protocolECKey().publicKeyAsHex
    }

    /**
     * (1) When you are creating a multi-sig wallet for yourself alone
     * as the genesis (wallet).
     * @param entranceFee the entrance fee you are sending
     * @return TransactionPackage
     */
    fun safeCreationAndSendGenesisWallet(
        entranceFee: Coin
    ): TransactionPackage {
        Log.i("Coin", "Coin: (safeCreationAndSendGenesisWallet start).")

        Log.i("Coin", "Coin: we are making a new genesis wallet for us alone.")
        val keys = listOf(ECKey.fromPublicOnly(networkPublicECKeyHex().hexToBytes()))
        val threshold = 1

        Log.i("Coin", "Coin: we will now make a ${keys.size}/${threshold} wallet")
        val transaction = Transaction(params)

        // Create the locking multi-sig script for the output.
        val script = ScriptBuilder.createMultiSigOutputScript(threshold, keys)

        // Add an output with the entrance fee & script.
        transaction.addOutput(entranceFee, script)

        Log.i("Coin", "Coin: your inputs will now be matched to entrance and fees.")
        val req = SendRequest.forTx(transaction)
        kit.wallet().completeTx(req)

        Log.i("Coin", "Coin: the change address is hard-reset to your protocol key.")
        req.changeAddress = Address.fromKey(params, protocolECKey(), Script.ScriptType.P2PKH)

        sendTransaction(req.tx)

        return TransactionPackage(
            req.tx.txId.toString(),
            "temp"
        )
    }

    /**
     * (2) Use this when you want to join an /existing/ wallet.
     * You need to broadcast this transaction to all the old owners so they can sign it.
     * @param networkPublicHexKeys list of NEW wallet owners
     * @param entranceFee the entrance fee
     * @param oldTransaction the old transaction
     * @param newThreshold the new threshold (default to # of new owners)
     * @return the resulting transaction (unsigned multi-sig input!)
     */
    fun safeCreationJoinWalletTransaction(
        networkPublicHexKeys: List<String>,
        entranceFee: Coin,
        oldTransaction: Transaction,
        newThreshold: Int = networkPublicHexKeys.size
    ): SendRequest {
        Log.i("Coin", "Coin: (safeCreationJoinWalletTransaction start).")

        Log.i("Coin", "Coin: making a transaction with you in it for everyone to sign.")
        val newTransaction = Transaction(params)
        val oldMultiSignatureOutput = getMultiSigOutput(oldTransaction).unsignedOutput

        Log.i("Coin", "Coin: output (1) -> we are adding the final new multi-sig output.")
        val newKeys = networkPublicHexKeys.map { publicHexKey: String ->
            Log.i("Coin", "Coin: de-serializing key ${publicHexKey}.")
            ECKey.fromPublicOnly(publicHexKey.hexToBytes())
        }
        val newMultiSignatureScript =
            ScriptBuilder.createMultiSigOutputScript(newThreshold, newKeys)

        // Calculate the final amount of coins (old coins + entrance fee) that will be the new multi-sig.
        val newMultiSignatureOutputMoney = oldMultiSignatureOutput.value.add(entranceFee)
        newTransaction.addOutput(newMultiSignatureOutputMoney, newMultiSignatureScript)

        Log.i("Coin", "Coin: input (1) -> we are adding the old multi-sig as input.")
        val multiSignatureInput = newTransaction.addInput(oldMultiSignatureOutput)
        // Disconnecting, because we will supply our own script_sig later (in signing process).
        multiSignatureInput.disconnect()

        Log.i("Coin", "Coin: use SendRequest to add our entranceFee inputs & change address.")
        val req = SendRequest.forTx(newTransaction)
        kit.wallet().completeTx(req)

        return req
    }

    /**
     * (2.1) You are (part) owner of a wallet a proposer wants to join. Sign the new wallet
     * and send it back to the proposer.
     * @param newTransaction the new transaction
     * @param oldTransaction the old transaction
     * @param key the key that will be signed with
     * @return the signature (you need to send back)
     */
    fun safeSigningJoinWalletTransaction(
        newTransaction: SendRequest,
        oldTransaction: Transaction,
        key: ECKey
    ): ECDSASignature {
        Log.i("Coin", "Coin: (safeSigningJoinWalletTransaction start).")

        val oldMultiSignatureOutput = getMultiSigOutput(oldTransaction).unsignedOutput
        val sighash: Sha256Hash = newTransaction.tx.hashForSignature(
            0,
            oldMultiSignatureOutput.scriptPubKey,
            Transaction.SigHash.ALL,
            false
        )
        val signature: ECDSASignature = key.sign(sighash)
        Log.i("Coin", "Coin: key -> ${key.publicKeyAsHex}")
        Log.i("Coin", "Coin: signature -> ${signature.encodeToDER().toHex()}")
        return signature
    }

    /**
     * (2.2) You are the proposer. You have collected the needed signatures and
     * will make the final transaction.
     * @param signaturesOfOldOwners signatures (of the OLD owners only, in correct order)
     * @param newTransaction SendRequest
     * @param oldTransaction Transaction
     * @return TransactionPackage?
     */
    fun safeSendingJoinWalletTransaction(
        signaturesOfOldOwners: List<ECDSASignature>,
        newTransaction: SendRequest,
        oldTransaction: Transaction
    ): TransactionPackage? {
        Log.i("Coin", "Coin: (safeSendingJoinWalletTransaction start).")
        val oldMultiSigOutput = getMultiSigOutput(oldTransaction).unsignedOutput

        Log.i("Coin", "Coin: make the new final transaction for the new wallet.")
        Log.i("Coin", "Coin: using ${signaturesOfOldOwners} signatures.")
        val transactionSignatures = signaturesOfOldOwners.map { signature ->
            TransactionSignature(signature, Transaction.SigHash.ALL, false)
        }
        val inputScript = ScriptBuilder.createMultiSigInputScript(transactionSignatures)

        // TODO: see if it is a issue to always assume the 1st input is the multi-sig input.
        val newMultiSigInput = newTransaction.tx.inputs[0]
        newMultiSigInput.scriptSig = inputScript

        // Verify the script before sending.
        try {
            newMultiSigInput.verify(oldMultiSigOutput)
            Log.i("Coin", "Coin: script is valid.")
        } catch (exception: VerificationException) {
            Log.i("Coin", "Coin: script is NOT valid. ${exception.message}.")
            return null
        }

        sendTransaction(newTransaction.tx)

        return TransactionPackage(
            newTransaction.tx.txId.toString(),
            "temp"
        )
    }

    /**
     * (3.1) There is a set-up multi-sig wallet and a proposal, create a signature
     * for the proposal.
     * @param transaction transaction with the multi-sig output
     * @param myPublicKey key to sign with (yourself most likely)
     * @param receiverAddress receiver address
     * @param value amount for receiver address
     * @return ECDSASignature
     */
    fun safeSigningTransactionFromMultiSig(
        transaction: Transaction,
        myPublicKey: ECKey,
        receiverAddress: Address,
        value: Coin
    ): ECDSASignature {
        Log.i("Coin", "Coin: (safeSigningTransactionFromMultiSig start).")

        Log.i("Coin", "Coin: a transaction will be signed from one of our multi-sig outputs.")
        // Retrieve the multi-signature contract.
        val multiSigOutput: TransactionOutput = getMultiSigOutput(transaction).unsignedOutput
        val multiSigScript: Script = multiSigOutput.scriptPubKey

        // Build the transaction we want to sign.
        val spendTx = Transaction(params)
        spendTx.addOutput(value, receiverAddress)
        spendTx.addInput(multiSigOutput)

        // Sign the transaction and return it.
        val sighash: Sha256Hash =
            spendTx.hashForSignature(0, multiSigScript, Transaction.SigHash.ALL, false)
        val signature: ECDSASignature = myPublicKey.sign(sighash)

        return signature
    }

    /**
     * (3.2) There is a set-up multi-sig wallet and there are enough signatures
     * to broadcast a transaction with.
     * @param transaction transaction with multi-sig output.
     * @param signatures signatures of owners (yourself included)
     * @param receiverAddress receiver address
     * @param value amount for receiver address
     * @return transaction
     */
    fun safeSendingTransactionFromMultiSig(
        transaction: Transaction,
        signatures: List<ECDSASignature>,
        receiverAddress: Address,
        value: Coin
    ): TransactionPackage? {
        Log.i("Coin", "Coin: (safeSendingTransactionFromMultiSig start).")

        // Retrieve the multi-sig output.
        val multiSigOutput: TransactionOutput = getMultiSigOutput(transaction).unsignedOutput

        Log.i("Coin", "Coin: making the transaction (again) that will be sent.")
        val spendTx = Transaction(params)
        spendTx.addOutput(value, receiverAddress)
        val input = spendTx.addInput(multiSigOutput)

        Log.i("Coin", "Coin: creating the input script to unlock the multi-sig input.")
        // Create the script that combines the signatures (to spend the multi-signature output).
        val transactionSignatures = signatures.map { signature ->
            TransactionSignature(signature, Transaction.SigHash.ALL, false)
        }
        val inputScript = ScriptBuilder.createMultiSigInputScript(transactionSignatures)
        // Set the script on the input.
        input.scriptSig = inputScript

        // Verify the script before sending.
        try {
            input.verify(multiSigOutput)
            Log.i("Coin", "Coin: script is valid.")
        } catch (exception: VerificationException) {
            Log.i("Coin", "Coin: script is NOT valid. ${exception.message}")
            return null
        }

        sendTransaction(spendTx)

        return TransactionPackage(
            spendTx.txId.toString(),
            spendTx.bitcoinSerialize().toHex()
        )
    }

    /**
     * Helper method to send transaction with logs and progress logs.
     * @param transaction transaction
     */
    private fun sendTransaction(transaction: Transaction) {
        Log.i("Coin", "Coin: (sendTransaction start).")
        Log.i("Coin", "Coin: txId: ${transaction.txId}")

        Log.i("Coin", "Coin: committing the transaction to our wallet.")
        kit.wallet().commitTx(transaction)

        val broadcastTransaction = kit.peerGroup().broadcastTransaction(transaction)
        broadcastTransaction.setProgressCallback { progress ->
            Log.i("Coin", "Coin: broadcast of transaction ${transaction.txId} progress: $progress.")
        }
        broadcastTransaction.broadcast()
        Log.i("Coin", "Coin: transaction broadcast of ${transaction.txId} is initiated.")
    }

    /**
     * Helper method to get the multi-sig output from a transaction.
     * NOTE: make sure that there is an actual multi-sig output!
     * @param transaction transaction with multi-sig output.
     * @return the multi-sig output
     */
    private fun getMultiSigOutput(transaction: Transaction): MultiSigOutputMeta {
        val multiSigOutputs = mutableListOf<TransactionOutput>()
        transaction.outputs.forEach { output ->
            if (ScriptPattern.isSentToMultisig(output.scriptPubKey)) {
                multiSigOutputs.add(output)
            }
        }

        if (multiSigOutputs.size != 1) {
            Log.i("Coin", "Coin: (getMultiSigOutput) the multi-sig output not available.")
        }

        val multiSigOutput = multiSigOutputs[0]

        return MultiSigOutputMeta(
            multiSigOutput.value,
            multiSigOutput.scriptPubKey.pubKeys,
            multiSigOutput.index,
            multiSigOutput.scriptPubKey.numberOfSignaturesRequiredToSpend,
            multiSigOutput
        )
    }

    /**
     * Helper method to attempt to get the transaction from a transaction ID
     * and return serialized version.
     * @param transactionId transactionId
     * @return null if not available in your wallet yet, else serialized version of transaction.
     */
    fun attemptToGetTransactionAndSerialize(transactionId: String): String? {
        val transaction = kit.wallet().getTransaction(Sha256Hash.wrap(transactionId))
        if (transaction != null) {
            val serializedTransaction = transaction.bitcoinSerialize().toHex()
            return serializedTransaction
        } else {
            Log.i(
                "Coin", "Coin: (attemptToGetTransactionAndSerialize) " +
                    "the transaction could not be found in your wallet."
            )
            return null
        }
    }

    /**
     * Helper method that prints useful information about a transaction.
     * @param transaction Transaction
     */
    fun printTransactionInformation(transaction: Transaction) {
        Log.i("Coin", "Coin: ============ Transaction Information ===============")
        Log.i("Coin", "Coin: txId ${transaction.txId}")
        Log.i("Coin", "Coin: fee ${transaction.fee}")
        Log.i("Coin", "Coin: inputs:::")
        transaction.inputs.forEach {
            Log.i("Coin", "Coin:    index ${it.index}")
            Log.i("Coin", "Coin:    value ${it.value}")
            Log.i("Coin", "Coin:    multi-sig ${ScriptPattern.isSentToMultisig(it.scriptSig)}")
        }
        Log.i("Coin", "Coin: outputs:::")
        transaction.outputs.forEach {
            Log.i("Coin", "Coin:    index ${it.index}")
            Log.i("Coin", "Coin:    value ${it.value}")
            Log.i("Coin", "Coin:    multi-sig ${ScriptPattern.isSentToMultisig(it.scriptPubKey)}")
        }
        Log.i("Coin", "Coin: multi-sig output::")
        val a = getMultiSigOutput(transaction)
        a.owners.forEach {
            Log.i("Coin", "Coin: key -> ${it.publicKeyAsHex}")
        }
        Log.i("Coin", "Coin:    # needed -> ${a.threshold}")
        Log.i("Coin", "Coin:    value -> ${a.value}")
        Log.i("Coin", "Coin: ============ Transaction Information ===============")
    }

    companion object {
        fun createMultiSignatureWallet(
            publicKeys: List<ECKey>,
            entranceFee: Coin,
            threshold: Int,
            params: NetworkParameters = MainNetParams.get()
        ): Transaction {
            // Prepare a template for the contract.
            val contract = Transaction(params)

            // Prepare a list of all keys present in contract.
            val keys = Collections.unmodifiableList(publicKeys)

            // Create a n-n multi-signature output script.
            val script = ScriptBuilder.createMultiSigOutputScript(threshold, keys)

            // Now add an output with the entrance fee & script.
            contract.addOutput(entranceFee, script)

            return contract
        }

        fun checkEntranceFeeTransaction(
            userBitcoinPk: Address,
            bitcoinTransactionHash: Sha256Hash,
            sharedWalletBitcoinPk: Address,
            entranceFee: Double
        ): Boolean {
            // Get transaction from tx hash
            val blockExplorer = BlockExplorer()
            val tx = try {
                blockExplorer.getTransaction(bitcoinTransactionHash.toString())
            } catch (e: Exception) {
                e.printStackTrace()
                return false
            }

            // Check block confirmations
            val blockHeightRelative = blockExplorer.latestBlock.height - tx.blockHeight
            if (blockHeightRelative < 6) {
                println("Transaction was not confirmed by at least 6 blocks:  $blockHeightRelative")
                return false
            }
            if (tx.blockHeight < 0) {
                println("Transaction does not have a valid block height: ${tx.blockHeight}")
                return false
            }

            // Check transaction inputs
            val userBitcoinPkString = userBitcoinPk.toString()
            var hasCorrectInput = false
            for (input in tx.inputs) {
                val inputValue = input.previousOutput.value.toDouble() / 100000000
                if (userBitcoinPkString.equals(input.previousOutput.address) &&
                    inputValue >= entranceFee
                ) {
                    hasCorrectInput = true
                    break
                }
            }

            if (!hasCorrectInput) {
                println("Transaction did not have correct inputs")
                return false
            }

            // Check transaction outputs
            val sharedWalletBitcoinPkString = sharedWalletBitcoinPk.toString()
            var hasCorrectOutput = false
            for (output in tx.outputs) {
                val outputValue = output.value.toDouble() / 100000000
                if (sharedWalletBitcoinPkString.equals(output.address) &&
                    outputValue >= entranceFee
                ) {
                    hasCorrectOutput = true
                    break
                }
            }

            if (!hasCorrectOutput) {
                println("Transaction did not have correct outputs")
                return false
            }

            return true
        }

        fun privateKeyStringToECKey(
            privateKey: String,
            params: NetworkParameters = MainNetParams.get()
        ): ECKey {
            return DumpedPrivateKey.fromBase58(params, privateKey).key
        }

        fun ecKeyToPrivateKeyString(
            ecKey: ECKey,
            params: NetworkParameters = MainNetParams.get()
        ): String {
            return ecKey.getPrivateKeyAsWiF(params)
        }

    }

    fun toSeed(): SerializedDeterminsticKey {
        val seed = kit.wallet().keyChainSeed
        val words = Joiner.on(" ").join(seed.mnemonicCode)
        val creationTime = seed.creationTimeSeconds
        return SerializedDeterminsticKey(words, creationTime)
    }

    data class TransactionPackage(
        val transactionId: String,
        val serializedTransaction: String
    )

    data class MultiSigOutputMeta(
        val value: Coin,
        val owners: MutableList<ECKey>,
        val index: Int,
        val threshold: Int,
        val unsignedOutput: TransactionOutput
    )

}
