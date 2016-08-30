package ai.coldbrew.karpathy;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DataSet {

    
    static Map<Integer, String> classLabels = new HashMap<>();
    static List<String> imagePaths;
    
    static String rootFolder = "root";
    static File train;
    static PrintWriter writer;
      
    
    public static List<String> getImagePaths() {
        return imagePaths;
    }
    
    public static Map<Integer, String> getClassLabels() {
        return classLabels;
    }
    
    public static void loadImagePaths(String folderPath) throws IOException {
    
        imagePaths = Files.readAllLines(Paths.get(folderPath + "train.txt"));
        List<String> labels = Files.readAllLines(Paths.get(folderPath + "labels.txt"));
        
        for(int i=0; i<labels.size(); i++)
            classLabels.put(i, labels.get(i));
    }
    
    public static void createImagePathsFile(String folderPath, String fileName) throws IOException {
    
        if (!folderPath.startsWith(rootFolder)) {
            rootFolder = folderPath;
            train = new File(rootFolder + fileName);
            writer = new PrintWriter(train);
        }
        
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();       

        for (File file : listOfFiles) {
            if (file.isFile()) {
                writer.println(file.getPath());                
            }
            else if (file.isDirectory()) {
                createImagePathsFile(file.getPath(), fileName);
            }
        }
        
        if (folderPath.equals(rootFolder))
            writer.close();
    }   
    
}
