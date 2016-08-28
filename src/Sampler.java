import java.io.FileReader;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import com.oracle.javafx.jmx.json.JSONDocument;
import org.ejml.data.DenseMatrix64F;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import umontreal.ssj.randvarmulti.DirichletGen;
import umontreal.ssj.probdistmulti.DirichletDist;
import umontreal.ssj.rng.*;


public class Sampler {

    private int nArticles;
    private int nAnnotators;
    private int nLabels;
    private int nTimes;
    private int nNewspapers;

    private static int firstYear = 1980;
    private static int nMonthsPerYear = 12;

    private HashMap<String, Integer> newspaperIndices;
    private HashMap<String, Integer> articleNameTime;
    private HashMap<String, Integer> articleNameNewspaper;
    private HashMap<Integer, Integer> articleTime;
    private HashMap<Integer, Integer> articleNewspaper;
    //private HashMap<Integer, ArrayList<Integer>> timeArticles;
    //private HashMap<Integer, ArrayList<Integer>> newspaperArticles;
    private HashMap<String, Integer> annotatorIndices;

    private ArrayList<String> articleNames;
    private ArrayList<HashMap<Integer, int[]>> annotations;

    private static double alpha = 1;
    private static double betaSigma = 0.1;
    private static double aMu = 5;
    private static double aSigma = 3;
    private static double bMu = 0;
    private static double bSigma = 3;
    private ArrayList<Double> betas;
    private ArrayList<DenseMatrix64F> timeFrames;     // phi
    private ArrayList<DenseMatrix64F> articleFrames;  // theta
    private double[][] a;
    private double[][] b;

    private static Random rand = new Random();
    private static RandomStream randomStream = new MRG32k3a();


    public Sampler(String inputFilename, String metadataFilename) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        Path metadataPath = Paths.get(metadataFilename);
        JSONObject metadata = (JSONObject) parser.parse(new FileReader(metadataPath.toString()));

        nLabels = 15;

        // index newspapers
        Set<String> newspapers = gatherNewspapers(metadata);
        nNewspapers = newspapers.size();
        newspaperIndices = new HashMap<>();
        int n = 0;
        for (String newspaper : newspapers) {
            newspaperIndices.put(newspaper, n++);
        }

        // record the time and paper of each article
        articleNameTime = new HashMap<>();
        articleNameNewspaper = new HashMap<>();
        nTimes = 0;
        for (Object articleName : metadata.keySet()) {
            JSONObject articleMetadata  = (JSONObject)metadata.get(articleName);
            String source = (String) articleMetadata.get("source");
            int year = ((Long) articleMetadata.get("year")).intValue();
            int month = ((Long) articleMetadata.get("month")).intValue() - 1;
            int time = (year - firstYear) * nMonthsPerYear + month;
            if (time >= 0) {
                articleNameTime.put(articleName.toString(), time);
                articleNameNewspaper.put(articleName.toString(), newspaperIndices.get(source));
            }
            if (time >= nTimes) {
                nTimes = time + 1;
            }

        }

        // index annotators
        Set<String> annotators = gatherAnnotators(data);
        nAnnotators = annotators.size();
        annotatorIndices = new HashMap<>();
        int j = 0;
        for (String annotator : annotators) {
            annotatorIndices.put(annotator, j++);
        }

        // read in the annotations and build up all relevant information
        articleNames = new ArrayList<>();
        articleTime = new HashMap<>();
        articleNewspaper = new HashMap<>();
        annotations = new ArrayList<>();
        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                JSONObject article = (JSONObject) data.get(articleName);
                JSONObject annotationsJson = (JSONObject) article.get("annotations");
                JSONObject framingAnnotations = (JSONObject) annotationsJson.get("framing");
                // make sure this article has annotations
                if (framingAnnotations.size() > 0) {
                    // get a new article id (i)
                    int i = articleNames.size();
                    // store the article name for future reference
                    articleNames.add(articleName.toString());
                    // get the timestep for this article (by name)
                    articleTime.put(i, articleNameTime.get(articleName.toString()));
                    // get the newspaper for this article (by name)
                    articleNewspaper.put(i, articleNameNewspaper.get(articleName.toString()));
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                    // loop through the annotators for this article
                    for (Object annotator : framingAnnotations.keySet()) {
                        // for each one, create an array to hold the annotations, and set to zero
                        int annotationArray[] = new int[nLabels];
                        // get the anntoator name and index
                        String parts[] = annotator.toString().split("_");
                        int annotatorIndex = annotatorIndices.get(parts[0]);
                        // loop through this annotator's annotations
                        JSONArray annotatorAnnotations = (JSONArray) framingAnnotations.get(annotator);
                        for (Object annotation : annotatorAnnotations) {
                            // get the code
                            double realCode = (Double) ((JSONObject) annotation).get("code");
                            // subtract 1 for zero-based indexing
                            int code = (int) Math.round(realCode) - 1;
                            // record this code as being present
                            annotationArray[code] = 1;
                        }
                        // store the annotations for this annotator
                        articleAnnotations.put(annotatorIndex, annotationArray);
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                }
            }
        }

        initialize();

    }


    private Set<String> gatherNewspapers(JSONObject metadata) {
        /*
        Read in the metadata and build up the set of newspapers for valid articles
         */
        Set <String> newspapers = new HashSet<>();
        for (Object key : metadata.keySet()) {
            JSONObject articleMetadata  = (JSONObject)metadata.get(key);
            String source = (String) articleMetadata.get("source");
            int year = ((Long) articleMetadata.get("year")).intValue();
            if (year >= firstYear) {
                newspapers.add(source);
            }
        }
        return newspapers;
    }

    private Set<String> gatherAnnotators(JSONObject data) {
        /*
        Read in the data and build up the set of annotators for valid articles
         */
        Set<String> annotators = new HashSet<>();

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                JSONObject article = (JSONObject) data.get(articleName);
                JSONObject annotations = (JSONObject) article.get("annotations");
                JSONObject framingAnnotations = (JSONObject) annotations.get("framing");
                for (Object annotator : framingAnnotations.keySet()) {
                    String parts[] = annotator.toString().split("_");
                    annotators.add(parts[0]);
                }
            }
        }
        return annotators;
    }

    /*
    Initialize random variables
     */
    private void initialize() {
        betas = new ArrayList<>();
        betas.add(Math.exp(rand.nextGaussian() * betaSigma));
        for (int t = 1; t < nTimes; t++) {
            betas.add(Math.exp(Math.log(betas.get(t-1)) + rand.nextGaussian() * betaSigma));
        }

        double alphas[] = new double[3];
        for (int q = 0; q < 3; q++) {
            alphas[q] = 0.5;
        }
        double p[] = new double[3];
        double p2[] = new double[3];
        DirichletGen.nextPoint(randomStream, alphas, p);
        DirichletGen.nextPoint(randomStream, alphas, p2);
        for (int q = 0; q < 3; q++) {
            System.out.println(p[q] + " " + p2[q]);
        }

        DirichletDist dirichletDist = new DirichletDist(alphas);
        double density = dirichletDist.density(p);
        System.out.println(density);

        density = dirichletDist.density(p2);
        System.out.println(density);

        density = dirichletDist.density(alphas);
        System.out.println(density);

    }

}
