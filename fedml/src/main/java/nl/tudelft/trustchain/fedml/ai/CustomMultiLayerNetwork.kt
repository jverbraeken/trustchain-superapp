package nl.tudelft.trustchain.fedml.ai


import mu.KotlinLogging
import org.deeplearning4j.exception.DL4JException
import org.deeplearning4j.nn.api.FwdPassType
import org.deeplearning4j.nn.api.layers.IOutputLayer
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.WorkspaceMode
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.workspace.ArrayType
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.nd4j.linalg.factory.Nd4j

private val logger = KotlinLogging.logger("CustomMultiLayerNetwork")

class CustomMultiLayerNetwork(conf: MultiLayerConfiguration) : MultiLayerNetwork(conf) {
    fun computeGradient() {
        if (outputLayer !is IOutputLayer) {
            throw DL4JException(
                "Cannot calculate gradient with respect to labels: final layer is not an IOutputLayer. " +
                    "Final layer class: " + outputLayer.javaClass + ". To calculate gradients and fit a network " +
                    "using backpropagation, the final layer must be an output layer")
        }

        val mgr: LayerWorkspaceMgr
        if (layerWiseConfigurations.trainingWorkspaceMode == WorkspaceMode.NONE) {
            mgr = LayerWorkspaceMgr.noWorkspaces()
        } else {
            mgr = LayerWorkspaceMgr.builder()
                .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                .build()
            if (layerWiseConfigurations.cacheMode != null) {
                //For now: store cache mode activations in activations workspace
                mgr.setWorkspace(ArrayType.FF_CACHE, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
            }
        }
        val tbptt = layerWiseConfigurations.backpropType == BackpropType.TruncatedBPTT
        val fwdType = if (tbptt) FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE else FwdPassType.STANDARD
        synchronizeIterEpochCounts()
        mgr.notifyScopeEntered(ArrayType.ACTIVATIONS).use { ws ->
            //First: do a feed-forward through the network
            //Note that we don't actually need to do the full forward pass through the output layer right now; but we do
            // need the input to the output layer to be set (such that backprop can be done)
            val activations =
                ffToLayerActivationsInWs(layers.size - 2, fwdType, tbptt, input, mask, null)
            if (!trainingListeners.isEmpty()) {
                //TODO: We possibly do want output layer activations in some cases here...
                for (tl in trainingListeners) {
                    tl.onForwardPass(this, activations)
                }
            }
            var inputToOutputLayer = activations[activations.size - 1]
            if (layerWiseConfigurations.getInputPreProcess(layers.size - 1) != null) {
                inputToOutputLayer = layerWiseConfigurations.getInputPreProcess(layers.size - 1)
                    .preProcess(inputToOutputLayer, inputMiniBatchSize, mgr)
                //Validate activations location
            }
            outputLayer.setInput(inputToOutputLayer, mgr)
            //Then: compute gradients
            val pair =
                calcBackpropGradients(null, true, false, false)
            gradient = pair?.first

            //Listeners
            if (!trainingListeners.isEmpty()) {
                Nd4j.getMemoryManager().scopeOutOfWorkspaces().use { workspace ->
                    for (tl in trainingListeners) {
                        tl.onBackwardPass(this)
                    }
                }
            }
        }

        //Clear the post noise/dropconnect parameters on the output layer
        outputLayer.clearNoiseWeightParams()
    }
}
