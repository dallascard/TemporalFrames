import java.io.FileReader;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import com.oracle.javafx.jmx.json.JSONDocument;
import org.ejml.data.DenseMatrix64F;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;


public class Sampler {

    private int nArticles;
    private int nAnnotators;
    private int nLabels;
    private int nTimes;

    private HashMap<String, Integer> annotatorIndices;
    private HashMap<Integer, Integer> articleTime;
    private HashMap<Integer, ArrayList<Integer>> timeArticles;
    private HashMap<Integer, ArrayList<Integer>> articleAnnotators;

    private ArrayList<String> articleNames;

    private int alpha;
    private ArrayList<Integer> beta;
    private ArrayList<DenseMatrix64F> timeFrames;
    private ArrayList<DenseMatrix64F> articleFrames;
    private int[][][] annotations;
    private double[][] a;
    private double[][] b;

    public Sampler(String inputFilename) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        //Path metadataPath = Paths.get(metadataFilename);
        //JSONArray metaData = (JSONArray) parser.parse(new FileReader(metadataPath.toString()));

        nArticles = data.size();
        nLabels = 15;

        // determine the set of annotators 
        Set<String> annotators = gatherAnnotators(data);
        nAnnotators = annotators.size();
        annotatorIndices = new HashMap<>();
        int j = 0;
        for (String annotator : annotators) {
            annotatorIndices.put(annotator, j++);
        }




    }

    private Set<String> gatherAnnotators(JSONObject data) {
        Set<String> annotators = new HashSet<>();

        for (Object articleName : data.keySet()) {
            JSONObject article = (JSONObject) data.get(articleName);
            JSONObject annotations = (JSONObject) article.get("annotations");
            JSONObject framingAnnotations = (JSONObject) annotations.get("framing");
            for (Object annotator : framingAnnotations.keySet()) {
                String parts[] = annotator.toString().split("_");
                annotators.add(parts[0]);
            }
        }
        return annotators;
    }

}
