package nl.tudelft.trustchain.fedml.ai.dataset

import java.io.File
import java.util.*
import kotlin.random.Random

class CustomFileSplit(datasetPath: File, random: Random, numSamplesPerLabel: Int, numLabels: Int) {
    val files: Array<out Array<File>>

    init {
        files = uriMap.computeIfAbsent(datasetPath.absolutePath) {
            listFiles(datasetPath, numSamplesPerLabel, numLabels).onEach { it.shuffle(random) }
        }
    }

    private fun listFiles(dir: File, numSamplesPerLabel: Int, numLabels: Int): Array<Array<File>> {
        val queue = LinkedList<File>()
        queue.add(dir)
        val placeholder = File("")
        val out = Array(numLabels) { Array(numSamplesPerLabel) { placeholder } }
        val count = IntArray(numLabels)
        while (!queue.isEmpty()) {
            val labelFile = queue.remove()
            val listFiles = labelFile.listFiles()
            if (listFiles != null) {
                val label = if (listFiles.first().isDirectory) -1 else labelFile.name.toInt()
                if (label < 6) {
                    for (f in listFiles) {
                        if (f.isDirectory) {
                            queue.add(f)
                        } else {
                            out[label][count[label]++] = f
                        }
                    }
                }
            }
        }
        return out
    }

    companion object {
        val uriMap = mutableMapOf<String, Array<out Array<File>>>()
    }
}
