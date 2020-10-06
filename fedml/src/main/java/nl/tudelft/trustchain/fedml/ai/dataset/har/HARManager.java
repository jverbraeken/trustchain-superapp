package nl.tudelft.trustchain.fedml.ai.dataset.har;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;


public class HARManager {
    private List<String>[] dataArr;
    private int[] labelsArr;

    public HARManager(File[] dataFiles, File labelsFile) throws IOException {
        dataArr = (List<String>[]) new List[dataFiles.length];
        for (int i = 0; i < dataFiles.length; i++) {
            try (BufferedReader reader = new BufferedReader(new FileReader(dataFiles[i]))) {
                dataArr[i] = reader.lines().collect(Collectors.toList());
            }
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(labelsFile))) {
            labelsArr = reader.lines().mapToInt(i -> Integer.parseInt(i) - 1).toArray();  // labels start at 1 instead of 0
        }
    }

    public String[] readEntryUnsafe(int i) {
        String[] res = new String[dataArr.length];
        for (int j = 0; j < dataArr.length; j++) {
            res[j] = dataArr[j].get(i);
        }
        return res;
    }

    public int readLabel(int i) {
        return labelsArr[i];
    }
}
