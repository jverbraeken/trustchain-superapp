package nl.tudelft.trustchain.fedml.ai.dataset

import nl.tudelft.trustchain.fedml.ai.dataset.cifar.CustomCifar10Fetcher
import java.io.File
import java.net.URI
import java.util.*
import kotlin.random.Random

class CustomFileSplit(datasetPath: File, random: Random, numSamplesPerLabel: Int) {
    val uris: Array<out Array<URI>>

    init {
        uris = uriMap.getOrPut(datasetPath.absolutePath) {
            listFiles(datasetPath, numSamplesPerLabel).onEach { it.shuffle(random) }
        }
    }

    private fun listFiles(dir: File, numSamplesPerLabel: Int): Array<Array<URI>> {
        val queue = LinkedList<File>()
        queue.add(dir)
        val placeholder = URI("")
        val out = Array(CustomCifar10Fetcher.NUM_LABELS) { Array(numSamplesPerLabel) { placeholder } }
        val count = IntArray(CustomCifar10Fetcher.NUM_LABELS)
        while (!queue.isEmpty()) {
            val labelFile = queue.remove()
            val listFiles = labelFile.listFiles()
            if (listFiles != null) {
                val label = if (listFiles.first().isDirectory) -1 else labelFile.name.toInt()
                for (f in listFiles) {
                    if (f.isDirectory) {
                        queue.add(f)
                    } else {
                        out[label][count[label]++] = f.toURI()
                    }
                }
            }
        }
        return out
    }

    companion object {
        val uriMap = mutableMapOf<String, Array<out Array<URI>>>()
    }
}