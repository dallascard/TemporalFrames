import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import umontreal.ssj.probdistmulti.DirichletDist;
import umontreal.ssj.rng.*;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.analysis.function.Sigmoid;


public class CombinedModel {

    private int nArticlesWithFraming;
    private int nArticlesWithTone;
    private int nFramingAnnotators;
    private int nToneAnnotators;
    private int nTimes;
    private int nNewspapers;

    private static int nLabels = 15;
    private static int nTones = 3;

    private static int firstYear = 1980;
    private static int firstQuarter = 0;
    private static int nMonthsPerYear = 12;
    private static int nQuartersPerYear = 4;
    private static int nMonthsPerQuarter = 3;
    private static int nFeatures = 7;

    private Set<String> irrelevantArticles;
    private Set<String> trainingArticles;
    private HashMap<String, Integer> newspaperIndices;
    private HashMap<String, Integer> articleNameTime;
    private HashMap<String, Integer> articleNameNewspaper;
    private HashMap<Integer, Integer> framingArticleTime;
    private HashMap<Integer, Integer> toneArticleTime;
    private HashMap<Integer, Integer> framingArticleNewspaper;
    private HashMap<Integer, Integer> toneArticleNewspaper;
    private HashMap<Integer, ArrayList<Integer>> timeFramingArticles;
    private HashMap<Integer, ArrayList<Integer>> timeToneArticles;
    private HashMap<String, Integer> framingAnnotatorIndices;
    private HashMap<String, Integer> toneAnnotatorIndices;
    private HashMap<Integer, ArrayList<Integer>> framingAnnotatorArticles;
    private HashMap<Integer, ArrayList<Integer>> toneAnnotatorArticles;

    private ArrayList<String> framingArticleNames;
    private ArrayList<String> toneArticleNames;
    private ArrayList<HashMap<Integer, int[]>> framingAnnotations;
    private ArrayList<HashMap<Integer, Integer>> toneAnnotations;

    private ArrayList<double[]> timeFramesReals;     // phi
    private ArrayList<double[]> timeFramesCube;
    private ArrayList<int[]> articleFrames ;  // theta
    private double[] timeEntropy;
    private ArrayList<double[]> timeToneReals;     // phi
    private ArrayList<double[]> timeToneSimplex;
    private int[] articleTone ;
    private double[][] q;  // framing sensitivity
    private double[][] r;  // framing specificity
    private double[][][] sSimplex;  // combined equivalent of sens / spec for tone
    private double[][][] sReal;
    private double[] weights;



    private double[] framesMean;
    private double[] tonesMean;
    private double[] globalMean;

    private double[] mood;
    private double[] nArticlesAtTime;

    private double timeFramesRealSigma = 0.01;
    private double timeToneRealSigma = 0.01;
    private double weightSigma = 5.0;
    private double moodSigma = 0.25;

    // Metropolis-Hastings step parameters
    private static double mhTimeFramesStepSigma = 0.05 ;
    private static double mhTimeFramesRealSigmaStep = 0.005;
    private static double mhTimeToneStepSigma = 0.1 ;
    private static double mhTimeToneRealSigmaStep = 0.01;
    private static double [] mhWeightsStepSigma = {0.05, 0.1, 0.2, 0.2, 0.5, 0.01, 0.05};
    private static double mhOneWeightStepSigma = 0.0001;
    private static double mhQSigma = 0.05;
    private static double mhRSigma = 0.02;
    private static double mhSSigma = 0.01;
    private static double [][] mhSCovariance = {{mhSSigma, 0, 0}, {0, mhSSigma, 0}, {0, 0, mhSSigma}};
    private static double mhMoodSigmaStep = 0.01;

    private static Random rand = new Random();
    private static RandomStream randomStream = new MRG32k3a();
    private static Sigmoid sigmoid = new Sigmoid();

    public CombinedModel(String inputFilename, String metadataFilename, String predictionsFilename, String moodFilename,
                         boolean normalizeStoriesAtTime, boolean normalizeMood) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        Path metadataPath = Paths.get(metadataFilename);
        JSONObject metadata = (JSONObject) parser.parse(new FileReader(metadataPath.toString()));

        // index newspapers
        Set<String> newspapers = gatherNewspapers(metadata);
        nNewspapers = newspapers.size();
        System.out.println(nNewspapers + " newspapers");
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
            //int quarter = (int) Math.floor((double) month / (double) nMonthsPerQuarter);
            //int time = (year - firstYear) * nQuartersPerYear + quarter;
            int time = yearAndMonthToTime(year, month);
            if (time >= 0) {
                articleNameTime.put(articleName.toString(), time);
                articleNameNewspaper.put(articleName.toString(), newspaperIndices.get(source));
            }
            if (time >= nTimes) {
                nTimes = time + 1;
            }
        }

        System.out.println(nTimes + " time periods");

        // record the total number of articles at each time step
        nArticlesAtTime = new double[nTimes];
        for (int time : articleNameTime.values()) {
            nArticlesAtTime[time] += 1;
        }
        // normalize nArticles
        double nMax = 0;
        for (int t = 0; t < nTimes; t++) {
            nMax = Math.max(nMax, nArticlesAtTime[t]);
        }
        if (normalizeStoriesAtTime) {
            for (int t = 0; t < nTimes; t++) {
                nArticlesAtTime[t] = nArticlesAtTime[t] / nMax;
            }
        }

        // intialize some empty arrays
        timeFramingArticles = new HashMap<>();
        for (int t = 0; t < nTimes; t++) {
            timeFramingArticles.put(t, new ArrayList<>());
        }
        timeToneArticles = new HashMap<>();
        for (int t = 0; t < nTimes; t++) {
            timeToneArticles.put(t, new ArrayList<>());
        }

        // determine the relevant articles based on annotations
        irrelevantArticles = getIrrelevantArticles(data);

        // index annotators
        Set<String> framingAnnotators = gatherFramingAnnotators(data);
        nFramingAnnotators= framingAnnotators.size();
        Set<String> toneAnnotators = gatherToneAnnotators(data);
        nToneAnnotators = toneAnnotators.size();

        System.out.println(nFramingAnnotators + " framing annotators");
        System.out.println(nToneAnnotators + " tone annotators");
        framingAnnotatorIndices = new HashMap<>();
        toneAnnotatorIndices = new HashMap<>();
        framingAnnotatorArticles = new HashMap<>();
        toneAnnotatorArticles = new HashMap<>();
        int k = 0;
        for (String annotator : framingAnnotators) {
            framingAnnotatorIndices.put(annotator, k);
            framingAnnotatorArticles.put(k, new ArrayList<>());
            k += 1;
        }
        System.out.println(framingAnnotatorIndices);
        k = 0;
        for (String annotator : toneAnnotators) {
            toneAnnotatorIndices.put(annotator, k);
            toneAnnotatorArticles.put(k, new ArrayList<>());
            k += 1;
        }
        System.out.println(toneAnnotatorIndices);

        // read in the annotations and build up all relevant information
        trainingArticles = new HashSet<>();
        framingArticleNames = new ArrayList<>();
        toneArticleNames = new ArrayList<>();
        framingArticleTime = new HashMap<>();
        toneArticleTime = new HashMap<>();
        framingArticleNewspaper = new HashMap<>();
        toneArticleNewspaper = new HashMap<>();
        framingAnnotations = new ArrayList<>();
        toneAnnotations = new ArrayList<>();
        framesMean = new double[nLabels];
        tonesMean = new double[nTones];

        int framingCount = 0;
        int toneCount = 0;
        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                if (!irrelevantArticles.contains(articleName.toString())) {
                    JSONObject article = (JSONObject) data.get(articleName);
                    JSONObject annotationsJson = (JSONObject) article.get("annotations");
                    JSONObject articleFramingAnnotations = (JSONObject) annotationsJson.get("framing");
                    JSONObject articleToneAnnotations = (JSONObject) annotationsJson.get("tone");
                    // make sure this article has annotations
                    if (articleFramingAnnotations.size() > 0) {
                        // get a new article id (i)
                        int i = framingArticleNames.size();
                        // store the article name for future reference
                        framingArticleNames.add(articleName.toString());
                        // get the timestep for this article (by name)
                        int time = articleNameTime.get(articleName.toString());
                        framingArticleTime.put(i, time);
                        // get the newspaper for this article (by name)
                        framingArticleNewspaper.put(i, articleNameNewspaper.get(articleName.toString()));
                        // create a hashmap to store the annotations for this article
                        HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                        // loop through the annotators for this article
                        for (Object annotator : articleFramingAnnotations.keySet()) {
                            // for each one, create an array to hold the annotations, and set to zero
                            int annotationArray[] = new int[nLabels];
                            // get the anntoator name and index
                            String parts[] = annotator.toString().split("_");
                            int annotatorIndex = framingAnnotatorIndices.get(parts[0]);
                            // loop through this annotator's annotations
                            JSONArray annotatorAnnotations = (JSONArray) articleFramingAnnotations.get(annotator);
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
                            framingAnnotatorArticles.get(annotatorIndex).add(i);
                            for (int j = 0; j < nLabels; j++) {
                                framesMean[j] += (double) annotationArray[j];
                            }
                            framingCount += 1;
                        }
                        // store the annotations for this article
                        framingAnnotations.add(articleAnnotations);
                        timeFramingArticles.get(time).add(i);
                    }
                    if (articleToneAnnotations.size() > 0) {
                        // get a new article id (i)
                        int i = toneArticleNames.size();
                        // store the article name for future reference
                        toneArticleNames.add(articleName.toString());
                        // get the timestep for this article (by name)
                        int time = articleNameTime.get(articleName.toString());
                        toneArticleTime.put(i, time);
                        // get the newspaper for this article (by name)
                        toneArticleNewspaper.put(i, articleNameNewspaper.get(articleName.toString()));
                        // create a hashmap to store the annotations for this article
                        HashMap<Integer, Integer> articleAnnotations = new HashMap<>();
                        // loop through the annotators for this article
                        for (Object annotator : articleToneAnnotations.keySet()) {
                            // for each one, prepare to hold the tone annotation
                            int toneAnnotation = 0;
                            // get the anntoator name and index
                            String parts[] = annotator.toString().split("_");
                            int annotatorIndex = toneAnnotatorIndices.get(parts[0]);
                            // loop through this annotator's annotations (should be only 1)
                            JSONArray annotatorAnnotations = (JSONArray) articleToneAnnotations.get(annotator);
                            for (Object annotation : annotatorAnnotations) {
                                // get the code
                                double realCode = (Double) ((JSONObject) annotation).get("code");
                                // subtract 17 for zero-based indexing
                                toneAnnotation = (int) Math.round(realCode) - 17;
                            }
                            // store the annotations for this annotator
                            articleAnnotations.put(annotatorIndex, toneAnnotation);
                            toneAnnotatorArticles.get(annotatorIndex).add(i);
                            tonesMean[toneAnnotation] += 1.0;
                            toneCount += 1;
                        }
                        // store the annotations for this article
                        toneAnnotations.add(articleAnnotations);
                        timeToneArticles.get(time).add(i);
                    }
                }
            }
        }

        // read in predictions and add to annotations
        Scanner scanner = new Scanner(new File(predictionsFilename));
        HashMap<String, int[]> framingPredictions = new HashMap<>();
        HashMap<String, Integer> tonePredictions = new HashMap<>();
        scanner.useDelimiter(",");
        // skip the header
        for (int i = 0; i < 19; i++) {
            String next = scanner.next();
            next = "";
        }
        String rowArticleName = "";
        int[] pArray = new int[nLabels];
        int tVal = 0;
        int iNext = 0;
        int isTrain = 0;
        while (scanner.hasNext()) {
            String next = scanner.next();
            int j = iNext % 19;
            //System.out.println(j + " " + next);
            if (j == 1) {
                rowArticleName = next;
            }
            if (j == 2) {
                if (Integer.parseInt(next) > 0) {
                    trainingArticles.add(rowArticleName);
                }
            }
            else if (j > 2 && j < 18) {
                pArray[j-3] = Integer.parseInt(next);
            }
            else if (j == 18) {
                tVal = Integer.parseInt(next);
                framingPredictions.put(rowArticleName, pArray);
                tonePredictions.put(rowArticleName, tVal);
                pArray = new int[nLabels];
            }
            iNext += 1;
        }

        //Do not forget to close the scanner
        scanner.close();

        // incorporate the predictions into the annotations
        int annotatorIndex = nFramingAnnotators;
        framingAnnotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : framingPredictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                if (!irrelevantArticles.contains(articleName)) {
                    //JSONObject article = (JSONObject) data.get(articleName);
                    if (framingArticleNames.contains(articleName)) {
                        if (!trainingArticles.contains(articleName)) {
                            // add the prediction to the set of new annotations
                            // get the article ID
                            int i = framingArticleNames.indexOf(articleName);
                            // create a hashmap to store the annotations for this article
                            HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(i);
                            // treat predictions as coming from a separate annotator
                            articleAnnotations.put(annotatorIndex, framingPredictions.get(articleName));
                            // store the annotations for this article
                            framingAnnotations.set(i, articleAnnotations);
                            // if this is not a training article, use it to estimate classifier properties
                            framingAnnotatorArticles.get(annotatorIndex).add(i);
                        }
                    } else {
                        // add information for a new article
                        // get a new article id (i)
                        int i = framingArticleNames.size();
                        // store the article name for future reference
                        framingArticleNames.add(articleName);
                        // get the timestep for this article (by name)
                        int time = articleNameTime.get(articleName);
                        framingArticleTime.put(i, time);
                        // get the newspaper for this article (by name)
                        framingArticleNewspaper.put(i, articleNameNewspaper.get(articleName));
                        // create a hashmap to store the annotations for this article
                        HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                        // treat predictions as coming from a separate anntoator
                        // loop through this annotator's annotations
                        articleAnnotations.put(annotatorIndex, framingPredictions.get(articleName));
                        // also use these unannotated articles for estimation of classifier properties
                        framingAnnotatorArticles.get(annotatorIndex).add(i);
                        //for (int j = 0; j < nLabels; j++) {
                        //    framesMean[j] += (double) framingPredictions.get(articleName)[j];
                        //}
                        // store the annotations for this article
                        framingAnnotations.add(articleAnnotations);
                        timeFramingArticles.get(time).add(i);
                        //framingCount += 1;
                    }
                }
            }
        }
        nFramingAnnotators += 1;

        System.out.println(framingAnnotations.size() + " articles with framing annotations");

        for (k = 0; k < nFramingAnnotators; k++) {
             System.out.println("Annotator: " + k + "; annotations: " + framingAnnotatorArticles.get(k).size());
        }

        /*
        // Try using all articles
        // treat the predictions as a new anntotator
        int annotatorIndex = nFramingAnnotators;
        framingAnnotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : framingPredictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                if (framingArticleNames.contains(articleName)) {
                    // get the article ID
                    int i = framingArticleNames.indexOf(articleName);
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(i);
                    // treat predictions as coming from a separate annotator
                    articleAnnotations.put(annotatorIndex, framingPredictions.get(articleName));
                    framingAnnotatorArticles.get(annotatorIndex).add(i);
                    // store the annotations for this article
                    framingAnnotations.set(i, articleAnnotations);
                }
            }
        }
        nFramingAnnotators += 1;
        */

        // treat the predictions as a new anntotator
        annotatorIndex = nToneAnnotators;
        toneAnnotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : tonePredictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                if (!irrelevantArticles.contains(articleName)) {
                    if (toneArticleNames.contains(articleName)) {
                        if (!trainingArticles.contains(articleName)) {
                            // get the article ID
                            int i = toneArticleNames.indexOf(articleName);
                            // create a hashmap to store the annotations for this article
                            HashMap<Integer, Integer> articleAnnotations = toneAnnotations.get(i);
                            // treat predictions as coming from a separate annotator
                            articleAnnotations.put(annotatorIndex, tonePredictions.get(articleName));
                            // store the annotations for this article
                            toneAnnotations.set(i, articleAnnotations);
                            // as above
                            toneAnnotatorArticles.get(annotatorIndex).add(i);
                        }
                    } else {
                        // add information for a new article
                        // get a new article id (i)
                        int i = toneArticleNames.size();
                        // store the article name for future reference
                        toneArticleNames.add(articleName);
                        // get the timestep for this article (by name)
                        int time = articleNameTime.get(articleName);
                        toneArticleTime.put(i, time);
                        // get the newspaper for this article (by name)
                        toneArticleNewspaper.put(i, articleNameNewspaper.get(articleName));
                        // create a hashmap to store the annotations for this article
                        HashMap<Integer, Integer> articleAnnotations = new HashMap<>();
                        // treat predictions as coming from a separate anntoator
                        // loop through this annotator's annotations
                        Integer tonePrediction = tonePredictions.get(articleName);
                        articleAnnotations.put(annotatorIndex, tonePrediction);
                        // as above
                        toneAnnotatorArticles.get(annotatorIndex).add(i);
                        //tonesMean[tonePrediction] += 1.0;
                        // store the annotations for this article
                        toneAnnotations.add(articleAnnotations);
                        timeToneArticles.get(time).add(i);
                        //toneCount += 1;
                    }
                }
            }
        }
        nToneAnnotators += 1;

        System.out.println(framingAnnotations.size() + " articles with tone annotations");

        // get the mean of the annotations as a sanity check
        for (int j = 0; j < nLabels; j++) {
            framesMean[j] = framesMean[j] / (double) framingCount;
        }
        for (int j = 0; j < nTones; j++) {
            tonesMean[j] = tonesMean[j] / (double) toneCount;
        }

        nArticlesWithFraming = framingAnnotations.size();
        nArticlesWithTone = toneAnnotations.size();

        System.out.println("Mean of annotations:");
        for (int j = 0; j < nLabels; j++) {
            System.out.print(framesMean[j] + " ");
        }

        System.out.println("Mean of tone annotations:");
        for (int j = 0; j < nTones; j++) {
            System.out.print(tonesMean[j] + " ");
        }
        System.out.println("");

        // read in mood data
        mood = new double[nTimes];
        scanner = new Scanner(new File(moodFilename));
        scanner.useDelimiter(",");
        // skip the header
        for (int i = 0; i < 6; i++) {
            String next = scanner.next();
        }
        int year = 0;
        int quarter = 0;
        double moodVal = 0;
        iNext = 0;
        while (scanner.hasNext()) {
            String next = scanner.next();
            int j = iNext % 6;
            if (j == 1) {
                year = Integer.parseInt(next);
            }
            else if (j == 2) {
                quarter = Integer.parseInt(next)-1;
            }
            else if (j == 3) {
                if (normalizeMood) {
                    // normalize mood to be in the range (-1, 1)
                    moodVal = Double.parseDouble(next) / 100.0;
                } else {
                    moodVal = Double.parseDouble(next);
                }

            }
            else if (j == 5) {
                int time = yearAndQuarterToTime(year, quarter);
                if (time >= 0 && time < nTimes) {
                    mood[time] = moodVal;
                }
            }
            iNext += 1;
        }

        //Do not forget to close the scanner
        scanner.close();

        // save mood data
        Path output_path;
        output_path = Paths.get("samples", "mood.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (int t = 0; t < nTimes; t++) {
                file.write(mood[t] + ",");
            }
        }


        // display mood data
        /*
        for (int t = 0; t < nTimes; t++) {
            //quarter = t % nQuartersPerYear;
            //year = (t - quarter) / nQuartersPerYear + firstYear;
            int nFramingArticles = timeFramingArticles.get(t).size();
            int nToneArticles = timeToneArticles.get(t).size();
            double yearQuarter = timeToYearAndQuarter(t);
            //System.out.println(yearQuarter + " : " + nFramingArticles + ", " + nToneArticles);
            //System.out.println(yearQuarter + " : " + mood[t]);
        }
        */

        initialize();

    }


    private int yearAndMonthToTime(int year, int month) {
        int quarter = (int) Math.floor((double) month / (double) nMonthsPerQuarter);
        return yearAndQuarterToTime(year, quarter);
    }

    private int yearAndQuarterToTime(int year, int quarter) {
        return (year - firstYear) * nQuartersPerYear + quarter - firstQuarter;
    }

    private double timeToYearAndQuarter(int time) {
        return (double) (time + firstQuarter) / (double) nQuartersPerYear + firstYear;
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

    private Set<String> getIrrelevantArticles(JSONObject data) {
        Set<String> irrelevantArticles = new HashSet<>();

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                boolean relevant = true;
                JSONObject article = (JSONObject) data.get(articleName);
                JSONObject annotations = (JSONObject) article.get("annotations");
                JSONObject relevanceJudgements = (JSONObject) annotations.get("irrelevant");
                for (Object annotator : relevanceJudgements.keySet()) {
                    boolean irrelevant = (boolean) relevanceJudgements.get(annotator);
                    if (irrelevant) {
                        relevant = false;
                    }
                }
                if (!relevant) {
                    irrelevantArticles.add(articleName.toString());
                }
            }
        }
        return irrelevantArticles;
    }



    private Set<String> gatherFramingAnnotators(JSONObject data) {
        /*
        Read in the data and build up the set of annotators for valid articles
         */
        Set<String> annotators = new HashSet<>();

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                if (!irrelevantArticles.contains(articleName.toString())) {
                    JSONObject article = (JSONObject) data.get(articleName);
                    JSONObject annotations = (JSONObject) article.get("annotations");
                    JSONObject framingAnnotations = (JSONObject) annotations.get("framing");
                    for (Object annotator : framingAnnotations.keySet()) {
                        String parts[] = annotator.toString().split("_");
                        annotators.add(parts[0]);
                    }
                }
            }
        }
        return annotators;
    }

    private Set<String> gatherToneAnnotators(JSONObject data) {
        /*
        Read in the data and build up the set of annotators for valid articles
         */
        Set<String> annotators = new HashSet<>();

        for (Object articleName : data.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName.toString())) {
                if (!irrelevantArticles.contains(articleName.toString())) {
                    JSONObject article = (JSONObject) data.get(articleName);
                    JSONObject annotations = (JSONObject) article.get("annotations");
                    JSONObject framingAnnotations = (JSONObject) annotations.get("tone");
                    for (Object annotator : framingAnnotations.keySet()) {
                        String parts[] = annotator.toString().split("_");
                        annotators.add(parts[0]);
                    }
                }
            }
        }
        return annotators;
    }

    private void initialize() {
        // initialize article frames based on annotations
        articleFrames = new ArrayList<>();
        for (int i = 0; i < nArticlesWithFraming; i++) {
            int initialLabels[] = new int[nLabels];
            double annotationCounts[] = new double[nLabels];
            // add the contributions of each annotator
            HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(i);
            double nAnnotators = (double) articleAnnotations.size();
            for (int annotator : articleAnnotations.keySet()) {
                int annotatorAnnotations[] = articleAnnotations.get(annotator);
                for (int j = 0; j < nLabels; j++) {
                    annotationCounts[j] += (double) annotatorAnnotations[j] / nAnnotators;
                }
            }
            for (int j = 0; j < nLabels; j++) {
                double u = rand.nextDouble();
                if (u < annotationCounts[j]) {
                    initialLabels[j] = 1;
                }
            }
            articleFrames.add(initialLabels);
        }

        // initialize article tones based on annotations
        articleTone = new int[nArticlesWithTone];
        for (int i = 0; i < nArticlesWithTone; i++) {
            double annotationCounts[] = new double[nTones];
            // add the contributions of each annotator
            HashMap<Integer, Integer> articleAnnotations = toneAnnotations.get(i);
            double nAnnotators = (double) articleAnnotations.size();
            for (int annotator : articleAnnotations.keySet()) {
                int annotatorTone = articleAnnotations.get(annotator);
                annotationCounts[annotatorTone] += 1.0 / nAnnotators;
            }
            MultinomialDistribution multinomialDistribution = new MultinomialDistribution(annotationCounts, nTones);
            int initialTone = multinomialDistribution.sample(rand);
            articleTone[i] = initialTone;
        }

        // initialize timeFrames as the mean of the corresponding articles (+ global mean)
        timeFramesReals = new ArrayList<>();
        timeFramesCube = new ArrayList<>();
        timeToneReals = new ArrayList<>();
        timeToneSimplex = new ArrayList<>();
        timeEntropy = new double[nTimes];

        for (int t = 0; t < nTimes; t++) {
            double timeFramesMean[] = new double[nLabels];
            double timeFramesMeanReals[] = new double[nLabels];
            // initialize everything with the global mean
            ArrayList<Integer> articles = timeFramingArticles.get(t);
            double nTimeFramingArticles = (double) articles.size();
            for (int j = 0; j < nLabels; j++) {
                timeFramesMean[j] += framesMean[j] / (nTimeFramingArticles + 1);
            }
            for (int i : articles) {
                int frames[] = articleFrames.get(i);
                for (int j = 0; j < nLabels; j++) {
                    timeFramesMean[j] += (double) frames[j] / (nTimeFramingArticles + 1);
                }
            }

            timeFramesMeanReals = Transformations.cubeToReals(timeFramesMean, nLabels);

            timeFramesCube.add(timeFramesMean);
            timeFramesReals.add(timeFramesMeanReals);
            timeEntropy[t] = computeEntropy(timeFramesMean);

            double timeTonesMeanSimplex[] = new double[nTones];

            // initialize everything with the global mean
            articles = timeToneArticles.get(t);
            double nTimeToneArticles = (double) articles.size();
            for (int j = 0; j < nTones; j++) {
                timeTonesMeanSimplex[j] += tonesMean[j] / (nTimeToneArticles + 1);
            }
            for (int i : articles) {
                int tone = articleTone[i];
                timeTonesMeanSimplex[tone] += 1.0 / (nTimeToneArticles + 1);
            }

            double timeTonesMeanReals[] = Transformations.simplexToReals(timeTonesMeanSimplex, nTones);

            timeToneSimplex.add(timeTonesMeanSimplex);
            timeToneReals.add(timeTonesMeanReals);

        }

        System.out.println("");

        // initialize annotator parameters to reasonable values
        q = new double[nFramingAnnotators][nLabels];
        r = new double[nFramingAnnotators][nLabels];
        for (int k = 0; k < nFramingAnnotators; k++) {
            for (int j = 0; j < nLabels; j++) {
                q[k][j] = 0.8;
                r[k][j] = 0.8;
            }
        }

        double priorCorrect = 0.8;
        sSimplex = new double[nToneAnnotators][nTones][nTones];
        sReal = new double[nToneAnnotators][nTones][nTones];
        for (int k = 0; k < nToneAnnotators; k++) {
            for (int j = 0; j < nTones; j++) {
                for (int j2 = 0; j2 < nTones; j2++) {
                    if (j == j2) {
                        sSimplex[k][j][j2] = priorCorrect;
                    } else {
                        sSimplex[k][j][j2] = (1.0 - priorCorrect) / (nTones - 1);
                    }
                }
                sReal[k][j] = Transformations.simplexToReals(sSimplex[k][j], nTones);
            }
        }

        // initialize weights
        weights = new double[nFeatures];
        /*
        weights[0] = 0.18;
        weights[1] = 0.82;
        weights[2] = 0.0;
        weights[3] = 2.0;
        weights[4] = 1.1;
        weights[5] = 0.0;
        weights[6] = -0.3;
        */

    }


    void run(int nIter, int burnIn, int samplingPeriod, int printPeriod) throws  Exception {
        int nSamples = (int) Math.floor((nIter - burnIn) / (double) samplingPeriod);
        System.out.println("Collecting " + nSamples + " samples");

        double timeFrameSamples [][][] = new double[nSamples][nTimes][nLabels];
        int articleFrameSamples [][][] = new int[nSamples][nArticlesWithFraming][nLabels];
        double timeFrameRealSigmaSamples [] = new double[nSamples];
        double timeToneSamples [][][] = new double[nSamples][nTimes][nLabels];
        int articleToneSamples [][] = new int[nSamples][nArticlesWithTone];
        double timeToneRealSigmaSamples [] = new double[nSamples];
        double weightSamples [][] = new double[nSamples][nFeatures];
        double qSamples [][][] = new double[nSamples][nFramingAnnotators][nLabels];
        double rSamples [][][] = new double[nSamples][nFramingAnnotators][nLabels];
        double sSamples [][][][] = new double[nSamples][nToneAnnotators][nTones][nTones];
        double entropySamples [][] = new double[nSamples][nTimes];
        double moodSamples[][] = new double[nSamples][nTimes];
        double moodSigmaSamples[] = new double[nSamples];

        double timeFrameRate = 0.0;
        double articleFramesRate = 0;
        double timeTonesRate = 0.0;
        double articleToneRates = 0.0;
        double timeFramesSigmaRate = 0;
        double timeToneSigmaRate = 0;
        double [] weightRate = new double[nFeatures];
        double oneWeightRate = 0.0;
        double [][] qRate = new double[nFramingAnnotators][nLabels];
        double [][] rRate = new double[nFramingAnnotators][nLabels];
        double [][] sRate = new double[nToneAnnotators][nTones];
        double moodSigmaRate = 0.0;

        int sample = 0;
        int i = 0;
        while (sample < nSamples) {
            timeFrameRate += sampleTimeFrames();
            sampleArticleFrames();
            timeFramesSigmaRate += sampleTimeFramesRealSigma();
            timeTonesRate += sampleTimeTones();
            sampleArticleTones();
            timeToneSigmaRate += sampleTimeToneRealSigma();
            double [] weightAcceptances = sampleWeights();
            for (int f = 0; f < nFeatures; f++) {
                weightRate[f] += (double) weightAcceptances[f];
            }
            //oneWeightRate += sampleAllWeights();

            //moodSigmaRate += sampleMoodSigma();

            double [][] qAcceptances = sampleQ();
            for (int k = 0; k < nFramingAnnotators; k++) {
                for (int j = 0; j < nLabels; j++) {
                    qRate[k][j] += qAcceptances[k][j];
                }
            }
            double [][] rAcceptances = sampleR();
            for (int k = 0; k < nFramingAnnotators; k++) {
                for (int j = 0; j < nLabels; j++) {
                    rRate[k][j] += rAcceptances[k][j];
                }
            }
            double [][] sAcceptances = sampleS();
            for (int k = 0; k < nToneAnnotators; k++) {
                for (int j = 0; j < nTones; j++) {
                    sRate[k][j] += sAcceptances[k][j];
                }
            }

            //globalMean = recomputeGlobalMean();

            // save samples

            if (i > burnIn && i % samplingPeriod == 0) {
                for (int t = 0; t < nTimes; t++) {
                    System.arraycopy(timeFramesCube.get(t), 0, timeFrameSamples[sample][t], 0, nLabels);
                }
                for (int t = 0; t < nTimes; t++) {
                    System.arraycopy(timeToneSimplex.get(t), 0, timeToneSamples[sample][t], 0, nTones);
                }
                timeFrameRealSigmaSamples[sample] = timeFramesRealSigma;
                timeToneRealSigmaSamples[sample] = timeToneRealSigma;
                for (int f = 0; f < nFeatures; f++) {
                    System.arraycopy(weights, 0, weightSamples[sample], 0, nFeatures);
                }
                /*
                for (int a = 0; a < nArticlesWithFraming; a++) {
                    System.arraycopy(articleFrames.get(a), 0, articleFrameSamples[sample][a], 0, nLabels);
                }
                System.arraycopy(articleTone, 0, articleToneSamples[sample], 0, nArticlesWithTone);
                */

                moodSigmaSamples[sample] = moodSigma;

                for (int k = 0; k < nFramingAnnotators; k++) {
                    System.arraycopy(q[k], 0, qSamples[sample][k], 0, nLabels);
                    System.arraycopy(r[k], 0, rSamples[sample][k], 0, nLabels);
                }
                for (int k = 0; k < nToneAnnotators; k++) {
                    for (int l = 0; l < nTones; l++) {
                        System.arraycopy(sSimplex[k][l], 0, sSamples[sample][k][l], 0, nTones);
                    }
                }
                System.arraycopy(timeEntropy, 0, entropySamples[sample], 0, nTimes);

                for (int t = 0; t < nTimes; t++) {
                    double [] featureVector = computeFeatureVector(t, timeToneSimplex.get(t), timeEntropy[t]);
                    double moodMean = 0.0;
                    for (int f = 0; f < nFeatures; f++) {
                        moodMean += featureVector[f] * weights[f];
                    }
                    moodSamples[sample][t] = moodMean;
                }

                sample += 1;
            }

            i += 1;
            if (i % printPeriod == 0) {
                System.out.print(i + ": ");
                for (int f = 0; f < nFeatures; f++) {
                    System.out.print(weights[f] + " ");
                }
                //for (int k = 0; k < nLabels; k++) {
                //    System.out.print(globalMean[k] + " ");
                //}
                System.out.print("\n");
            }

        }

        // Display acceptance rates
        System.out.println(timeFrameRate / i);
        System.out.println(timeFramesSigmaRate / i);
        System.out.println(timeTonesRate / i);
        System.out.println(timeToneSigmaRate / i);
        System.out.println(moodSigmaRate / i);
        System.out.println("weight rates");
        for (int f = 0; f < nFeatures; f++) {
            System.out.println(weightRate[f] / i);
        }
        //System.out.println("weight rates: " + oneWeightRate / i);
        System.out.println(articleFramesRate / i);
        System.out.println(articleToneRates / i);
        System.out.println("Q rates");
        for (int k = 0; k < nFramingAnnotators; k++) {
            for (int j = 0; j < nLabels; j++) {
                System.out.print(qRate[k][j] / i + " ");
            }
            System.out.print("\n");
        }
        System.out.println("R rates");
        for (int k = 0; k < nFramingAnnotators; k++) {
            for (int j = 0; j < nLabels; j++) {
                System.out.print(rRate[k][j] / i + " ");
            }
            System.out.print("\n");
        }
        System.out.println("S rates");
        for (int k = 0; k < nToneAnnotators; k++) {
            for (int j = 0; j < nTones; j++) {
                System.out.print(sRate[k][j] / i + " ");
            }
            System.out.print("\n");
        }


        // save results
        Path output_path;
        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "timeFramesSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (sample = 0; sample < nSamples; sample++) {
                    for (int t = 0; t < nTimes; t++) {
                        file.write(timeFrameSamples[sample][t][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        output_path = Paths.get("samples", "timeFrameRealSigmaSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (sample = 0; sample < nSamples; sample++) {
                file.write(timeFrameRealSigmaSamples[sample] + ",\n" );
            }
        }


        for (int k = 0; k < nTones; k++) {
            output_path = Paths.get("samples", "timeToneSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (sample = 0; sample < nSamples; sample++) {
                    for (int t = 0; t < nTimes; t++) {
                        file.write(timeToneSamples[sample][t][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        output_path = Paths.get("samples", "timeToneRealSigmaSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (sample = 0; sample < nSamples; sample++) {
                file.write(timeToneRealSigmaSamples[sample] + ",\n");
            }
        }

        output_path = Paths.get("samples", "entropySamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (sample = 0; sample < nSamples; sample++) {
                for (int t = 0; t < nTimes; t++) {
                    file.write(entropySamples[sample][t] + ",");
                }
                file.write("\n");
            }
        }

        output_path = Paths.get("samples", "weightSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (sample = 0; sample < nSamples; sample++) {
                for (int f = 0; f < nFeatures; f++) {
                    file.write(weightSamples[sample][f] + ",");
                }
                file.write("\n");
            }
        }

        for (int j = 0; j < nLabels; j++) {
            output_path = Paths.get("samples", "qSamples" + j + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (sample = 0; sample < nSamples; sample++) {
                    for (int k = 0; k < nFramingAnnotators; k++) {
                        file.write(qSamples[sample][k][j] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        for (int j = 0; j < nLabels; j++) {
            output_path = Paths.get("samples", "rSamples" + j + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (sample = 0; sample < nSamples; sample++) {
                    for (int k = 0; k < nFramingAnnotators; k++) {
                        file.write(rSamples[sample][k][j] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        for (int j = 0; j < nTones; j++) {
            output_path = Paths.get("samples", "sSensitivitySamples" + j + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (sample = 0; sample < nSamples; sample++) {
                    for (int k = 0; k < nToneAnnotators; k++) {
                        file.write(sSamples[sample][k][j][j] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        output_path = Paths.get("samples", "moodSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (sample = 0; sample < nSamples; sample++) {
                for (int t = 0; t < nTimes; t++) {
                    file.write(moodSamples[sample][t] + ",");
                }
                file.write("\n");
            }
        }

        // What I actually want is maybe the annotation probabilities (?)

        /*
        output_path = Paths.get("samples", "articleMeans.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (int a = 0; a < nArticles; a++) {
                double mean [] = new double[nLabels];
                HashMap<Integer, int[]> articleAnnotations = annotations.get(a);
                for (int annotator : articleAnnotations.keySet()) {
                    for (int k = 0; k < nLabels; k++) {
                        mean[k] += (double) articleAnnotations.get(annotator)[k] / (double) articleAnnotations.size();
                    }
                }
                for (int k = 0; k < nLabels; k++) {
                    file.write(mean[k] + ",");
                }
                file.write("\n");
            }
        }
        */

        /*
        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "articleProbs" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int a = 0; a < nArticles; a++) {
                        double pk = sigmoid.value(articleFrameSamples[s][a][k] * zealSamples[s][0][k] - biasSamples[s][0][k]);
                        file.write(pk + ",");
                    }
                    file.write("\n");
                }
            }
        }
        */
    }

    private double[] computeFeatureVector(int time, double [] toneProbs, double entropy) {
        double featureVector[] = new double[nFeatures];
        featureVector[0] = 1;                                    // intercept
        if (time > 0) {
            featureVector[1] = mood[time - 1];                   // mood at t-1
        }
        else {
            featureVector[1] = mood[time];                       // cheap substitute for previous mood in first year
        }
        featureVector[2] = nArticlesAtTime[time];                // number of articles published in time t
        featureVector[3] = toneProbs[0] - toneProbs[2];          // net tone at time t
        featureVector[4] = featureVector[2] * featureVector[3];  // interaction b/w tone and nArticles
        featureVector[5] = entropy;                              // entropy
        featureVector[6] = featureVector[3] * featureVector[5];  // interaction b/w tone and entropy

        return featureVector;
    }

    private double computeEntropy(double [] framingProbs) {
        double entropy = 0;
        for (int j = 0; j < nLabels; j++) {
            entropy -= (framingProbs[j] * Math.log(framingProbs[j]) + (1-framingProbs[j]) * Math.log(1-framingProbs[j]));
        }
        return entropy;
    }

    private double computeLogProbMood(double [] featureVector, double [] weights, double mood, double moodSigma) {
        double mean = 0;
        for (int f = 0; f < nFeatures; f++) {
            mean += weights[f] * featureVector[f];
        }
        NormalDistribution currentMoodDist = new NormalDistribution(mean, moodSigma);
        double pLogMood = Math.log(currentMoodDist.density(mood));

        return pLogMood;
    }


    private double sampleTimeFrames() {
        // run the distribution over frames for the first time point

        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            // get the current distribution over frames
            double currentCube[] = timeFramesCube.get(t);
            double currentReals[] = timeFramesReals.get(t);
            // create a variable for a proposal
            double proposalReals[] = new double[nLabels];

            double mean[] = new double[nLabels];
            double covariance[][] = new double[nLabels][nLabels];
            for (int k = 0; k < nLabels; k++) {
                covariance[k][k] = 1;
            }
            MultivariateNormalDistribution normDist = new MultivariateNormalDistribution(mean, covariance);
            double normalStep[] = normDist.sample();
            double step = rand.nextGaussian() * mhTimeFramesStepSigma;

            // apply the step to generate a proposal
            for (int k = 0; k < nLabels; k++) {
                proposalReals[k] = currentReals[k] + normalStep[k] * step;
            }

            // transform the proposal to the cube
            double proposalCube[] = Transformations.realsToCube(proposalReals, nLabels);

            // get the distribution over frames at the previous time point
            double previousReals[] = new double[nLabels];
            double priorCovariance[][] = new double[nLabels][nLabels];
            double priorNextCovariance[][] = new double[nLabels][nLabels];

            if (t > 0) {
                previousReals = timeFramesReals.get(t-1);
                for (int k = 0; k < nLabels; k++) {
                    priorCovariance[k][k] = timeFramesRealSigma;
                    priorNextCovariance[k][k] = timeFramesRealSigma;
                }
            } else {
                // if t == 0, use the global mean
                //double [] previousCube = new double[nLabels];
                //System.arraycopy(framesMean, 0, previousCube, 0, nLabels);
                //previousReals = Transformations.cubeToReals(previousCube, nLabels);
                for (int k = 0; k < nLabels; k++) {
                    priorCovariance[k][k] = timeFramesRealSigma * 100;
                    priorNextCovariance[k][k] = timeFramesRealSigma;
                }
            }

            MultivariateNormalDistribution previousDist = new MultivariateNormalDistribution(previousReals, priorCovariance);

            double pLogCurrent = Math.log(previousDist.density(currentReals));
            double pLogProposal = Math.log(previousDist.density(proposalReals));

            // get the distribution over frames in the next time point
            if (t < nTimes-1) {
                double nextReals[] = timeFramesReals.get(t+1);

                // compute a distribution over a new distribution over frames for the current distribution
                MultivariateNormalDistribution currentDist = new MultivariateNormalDistribution(currentReals, priorNextCovariance);
                MultivariateNormalDistribution proposalDist = new MultivariateNormalDistribution(proposalReals, priorNextCovariance);

                pLogCurrent += Math.log(currentDist.density(nextReals));
                pLogProposal += Math.log(proposalDist.density(nextReals));
            }

            // compute the probability of the article distributions for both current and proposal
            ArrayList<Integer> articles = timeFramingArticles.get(t);
            for (int i : articles) {
                int articleLabels[] = articleFrames.get(i);
                for (int k = 0; k < nLabels; k++) {
                    pLogCurrent += articleLabels[k] * Math.log(currentCube[k]) + (1-articleLabels[k]) * Math.log(1-currentCube[k]);
                    pLogProposal += articleLabels[k] * Math.log(proposalCube[k]) + (1-articleLabels[k]) * Math.log(1-proposalCube[k]);
                }
            }

            // compute probabilities of mood for current and proposed latent framing probs
            double currentVector[] = computeFeatureVector(t, timeToneSimplex.get(t), timeEntropy[t]);
            double proposalEntropy = computeEntropy(proposalCube);
            double proposalVector[] = computeFeatureVector(t, timeToneSimplex.get(t), proposalEntropy);

            pLogCurrent += computeLogProbMood(currentVector, weights, mood[t], moodSigma);
            pLogProposal += computeLogProbMood(proposalVector, weights, mood[t], moodSigma);


            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFramesCube.set(t, proposalCube);
                timeFramesReals.set(t, proposalReals);
                nAccepted += 1;
                timeEntropy[t] = proposalEntropy;
            }

        }
        return nAccepted / nTimes;
    }

    private void sampleArticleFrames() {
        // don't bother to track acceptance because we're going to properly use Gibbs for this

        // loop through all articles
        for (int i = 0; i < nArticlesWithFraming; i++) {

            // get the current article dist
            int articleLabels[] = articleFrames.get(i);
            int proposedLabels[] = new int[nLabels];

            int time = framingArticleTime.get(i);
            double timeFrames[] = timeFramesCube.get(time);

            for (int j = 0; j < nLabels; j++) {

                double pLogPosGivenTime = Math.log(timeFrames[j]);
                double pLogNegGivenTime = Math.log(1-timeFrames[j]);

                // compute the probability of the labels for both current and proposal
                HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(i);
                for (int annotator : articleAnnotations.keySet()) {
                    int labels[] = articleAnnotations.get(annotator);
                    pLogPosGivenTime += labels[j] * Math.log(q[annotator][j]) + (1-labels[j]) * Math.log(1-r[annotator][j]);
                    pLogNegGivenTime += labels[j] * Math.log(1-q[annotator][j]) + (1-labels[j]) * Math.log(r[annotator][j]);
                }

                double pPosUnnorm = Math.exp(pLogPosGivenTime);
                double pNegUnnorm = Math.exp(pLogNegGivenTime);
                double pPos = pPosUnnorm / (pPosUnnorm + pNegUnnorm);

                double u = rand.nextDouble();

                if (u < pPos) {
                    proposedLabels[j] = 1;
                }
            }
            articleFrames.set(i, proposedLabels);
        }
    }

    private double sampleTimeFramesRealSigma() {
        double acceptance = 0.0;

        double current = timeFramesRealSigma;
        double proposal = current + rand.nextGaussian() * mhTimeFramesRealSigmaStep;

        double pLogCurrent = 0.0;
        double pLogProposal = 0.0;

        double [][] currentCovar = new double[nLabels][nLabels];
        double [][] proposalCovar = new double[nLabels][nLabels];

        for (int j = 0; j < nLabels; j++) {
            currentCovar[j][j] = current;
            proposalCovar[j][j] = proposal;
        }

        if (proposal > 0) {
            for (int t = 1; t < nTimes; t++) {
                MultivariateNormalDistribution priorCurrent = new MultivariateNormalDistribution(timeFramesReals.get(t-1), currentCovar);
                MultivariateNormalDistribution priorProposal = new MultivariateNormalDistribution(timeFramesReals.get(t-1), proposalCovar);

                pLogCurrent += Math.log(priorCurrent.density(timeFramesReals.get(t)));
                pLogProposal += Math.log(priorProposal.density(timeFramesReals.get(t)));
            }
            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFramesRealSigma = proposal;
                acceptance += 1;
            }
        }

        return acceptance;
    }

    private double sampleTimeToneRealSigma() {
        double acceptance = 0.0;

        double current = timeToneRealSigma;
        double proposal = current + rand.nextGaussian() * mhTimeToneRealSigmaStep;

        double pLogCurrent = 0.0;
        double pLogProposal = 0.0;

        double [][] currentCovar = new double[nTones][nTones];
        double [][] proposalCovar = new double[nTones][nTones];

        for (int j = 0; j < nTones; j++) {
            currentCovar[j][j] = current;
            proposalCovar[j][j] = proposal;
        }

        if (proposal > 0) {
            for (int t = 1; t < nTimes; t++) {
                MultivariateNormalDistribution priorCurrent = new MultivariateNormalDistribution(timeToneReals.get(t-1), currentCovar);
                MultivariateNormalDistribution priorProposal = new MultivariateNormalDistribution(timeToneReals.get(t-1), proposalCovar);

                pLogCurrent += Math.log(priorCurrent.density(timeToneReals.get(t)));
                pLogProposal += Math.log(priorProposal.density(timeToneReals.get(t)));
            }
            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeToneRealSigma = proposal;
                acceptance += 1;
            }
        }

        return acceptance;
    }


    private double sampleTimeTones() {
        // run the distribution over tones for the first time point

        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            // get the current distribution over tones
            double currentSimplex[] = timeToneSimplex.get(t);
            double currentReals[] = timeToneReals.get(t);
            // create a variable for a proposal
            double proposalReals[] = new double[nTones];

            double mean[] = new double[nTones];
            double covariance[][] = new double[nTones][nTones];
            for (int k = 0; k < nTones; k++) {
                covariance[k][k] = 1;
            }
            MultivariateNormalDistribution normDist = new MultivariateNormalDistribution(mean, covariance);
            double normalStep[] = normDist.sample();
            double step = rand.nextGaussian() * mhTimeToneStepSigma;

            // apply the step to generate a proposal
            for (int k = 0; k < nTones; k++) {
                proposalReals[k] = currentReals[k] + normalStep[k] * step;
            }

            // transform the proposal to the simplex
            double proposalSimplex[] = Transformations.realsToSimplex(proposalReals, nTones);

            // get the distribution over tones at the previous time point
            double previousReals[] = new double[nTones];
            double priorCovariance[][] = new double[nTones][nTones];
            double priorNextCovariance[][] = new double[nTones][nTones];
            if (t > 0) {
                previousReals = timeToneReals.get(t-1);
                // compute a distribution over the current time point given previous
                for (int k = 0; k < nTones; k++) {
                    priorCovariance[k][k] = timeToneRealSigma;
                    priorNextCovariance[k][k] = timeToneRealSigma;
                }
            } else {
                // if t == 0, use the global mean
                //double [] previousSimplex = new double[nTones];
                //System.arraycopy(tonesMean, 0, previousSimplex, 0, nTones);
                //previousReals = Transformations.simplexToReals(previousSimplex, nTones);
                // compute a distribution over the current time point given previous
                for (int k = 0; k < nTones; k++) {
                    priorCovariance[k][k] = timeToneRealSigma * 100;
                    priorNextCovariance[k][k] = timeToneRealSigma;
                }
            }


            MultivariateNormalDistribution previousDist = new MultivariateNormalDistribution(previousReals, priorCovariance);

            double pLogCurrent = Math.log(previousDist.density(currentReals));
            double pLogProposal = Math.log(previousDist.density(proposalReals));

            // get the distribution over tones in the next time point
            if (t < nTimes-1) {
                double nextReals[] = timeToneReals.get(t+1);

                // compute a distribution over a new distribution over tones for the current distribution
                MultivariateNormalDistribution currentDist = new MultivariateNormalDistribution(currentReals, priorNextCovariance);
                MultivariateNormalDistribution proposalDist = new MultivariateNormalDistribution(proposalReals, priorNextCovariance);

                pLogCurrent += Math.log(currentDist.density(nextReals));
                pLogProposal += Math.log(proposalDist.density(nextReals));
            }

            // compute the probability of the article tones for both current and proposal
            ArrayList<Integer> articles = timeToneArticles.get(t);
            for (int i : articles) {
                int iTone = articleTone[i];
                pLogCurrent += Math.log(currentSimplex[iTone]);
                pLogProposal += Math.log(proposalSimplex[iTone]);
            }

            // compute probabilities of mood for current and proposed latent framing probs
            double currentVector[] = computeFeatureVector(t, currentSimplex, timeEntropy[t]);
            double proposalVector[] = computeFeatureVector(t, proposalSimplex, timeEntropy[t]);

            pLogCurrent += computeLogProbMood(currentVector, weights, mood[t], moodSigma);
            pLogProposal += computeLogProbMood(proposalVector, weights, mood[t], moodSigma);

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeToneSimplex.set(t, proposalSimplex);
                timeToneReals.set(t, proposalReals);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }


    private void sampleArticleTones() {
        // don't bother to track acceptance because we're going to properly use Gibbs for this

        // loop through all articles
        for (int i = 0; i < nArticlesWithTone; i++) {

            // get the current article dist
            int tone = articleTone[i];

            int time = toneArticleTime.get(i);
            double timeTones[] = timeToneSimplex.get(time);

            double pLogPositive = Math.log(timeTones[0]);
            double pLogNeutral = Math.log(timeTones[1]);
            double pLogAnti = Math.log(timeTones[2]);

            HashMap<Integer, Integer> articleAnnotations = toneAnnotations.get(i);
            for (int annotator : articleAnnotations.keySet()) {
                int annotatorTone = articleAnnotations.get(annotator);
                pLogPositive += Math.log(sSimplex[annotator][0][annotatorTone]);
                pLogNeutral += Math.log(sSimplex[annotator][1][annotatorTone]);
                pLogAnti += Math.log(sSimplex[annotator][2][annotatorTone]);
            }

            double pPositiveUnnorm = Math.exp(pLogPositive);
            double pNeutralUnnorm = Math.exp(pLogNeutral);
            double pAntiUnnorm = Math.exp(pLogAnti);
            double pPos = pPositiveUnnorm / (pPositiveUnnorm + pNeutralUnnorm + pAntiUnnorm);
            double pProNeutral = pPos + pNeutralUnnorm / (pPositiveUnnorm + pNeutralUnnorm + pAntiUnnorm);

            double u = rand.nextDouble();

            if (u < pPos) {
                articleTone[i] = 0;
            }
            else if (u < pProNeutral) {
                articleTone[i] = 1;
            }
            else {
                articleTone[i] = 2;
            }

        }
    }


    private double [] sampleWeights() {
        double [] nAccepted = new double[nFeatures];

        for (int f = 0; f < nFeatures; f++) {
            double [] current = new double[nFeatures];
            double [] proposal = new double[nFeatures];
            System.arraycopy(weights, 0, current, 0, nFeatures);
            System.arraycopy(weights, 0, proposal, 0, nFeatures);

            double currentVal = current[f];
            double proposalVal = currentVal + rand.nextGaussian() * mhWeightsStepSigma[f];
            proposal[f] = proposalVal;

            NormalDistribution prior = new NormalDistribution(0, weightSigma);
            double pLogCurrent = Math.log(prior.density(currentVal));
            double pLogProposal = Math.log(prior.density(proposalVal));

            for (int t = 0; t < nTimes; t++) {
                double [] featureVector = computeFeatureVector(t, timeToneSimplex.get(t), timeEntropy[t]);
                pLogCurrent += computeLogProbMood(featureVector, current, mood[t], moodSigma);
                pLogProposal += computeLogProbMood(featureVector, proposal, mood[t], moodSigma);
            }
            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                weights[f] = proposalVal;
                nAccepted[f] += 1;
            }

            if (Double.isInfinite(pLogProposal)) {
                System.out.println("Inf in sample weight " + f);
            }

        }
        return nAccepted;
    }


    private double sampleAllWeights() {
        double nAccepted = 0.0;

        double [] current = new double[nFeatures];
        double [] proposal = new double[nFeatures];
        System.arraycopy(weights, 0, current, 0, nFeatures);
        //System.arraycopy(weights, 0, proposal, 0, nFeatures);

        //double currentVal = current[f];
        double [][] covar = new double [nFeatures][nFeatures];
        for (int f = 0; f < nFeatures; f++) {
            covar[f][f] = mhOneWeightStepSigma;
        }
        MultivariateNormalDistribution proposalDist = new MultivariateNormalDistribution(current, covar);

        proposal = proposalDist.sample();

        /*
        double [] means = new double[nFeatures];
        for (int f = 0; f < nFeatures; f++) {
            covar[f][f] = weightSigma;
        }
        */
        NormalDistribution prior = new NormalDistribution(0, weightSigma);
        double pLogCurrent = 0.0;
        double pLogProposal = 0.0;

        for (int f = 0; f < nFeatures; f++) {
            pLogCurrent += Math.log(prior.density(current[f]));
            pLogProposal += Math.log(prior.density(proposal[f]));
        }

        for (int t = 0; t < nTimes; t++) {
            double [] featureVector = computeFeatureVector(t, timeToneSimplex.get(t), timeEntropy[t]);
            pLogCurrent += computeLogProbMood(featureVector, current, mood[t], moodSigma);
            pLogProposal += computeLogProbMood(featureVector, proposal, mood[t], moodSigma);
        }

        double a = Math.exp(pLogProposal - pLogCurrent);
        double u = rand.nextDouble();

        if (u < a) {
            System.arraycopy(proposal, 0, weights, 0, nFeatures);
            nAccepted += 1;
        }

        return nAccepted;
    }



    private double sampleMoodSigma() {
        double acceptance = 0.0;

        double current = moodSigma;
        double proposal = current + rand.nextGaussian() * mhMoodSigmaStep;

        double pLogCurrent = 0.0;
        double pLogProposal = 0.0;

        if (proposal > 0) {
            for (int t = 1; t < nTimes; t++) {

                double [] featureVector = computeFeatureVector(t, timeToneSimplex.get(t), timeEntropy[t]);

                pLogCurrent += Math.log(computeLogProbMood(featureVector, weights, mood[t], current));
                pLogProposal += Math.log(computeLogProbMood(featureVector, weights, mood[t], proposal));
            }
            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFramesRealSigma = proposal;
                acceptance += 1;
            }
        }

        return acceptance;
    }


    double [][] sampleQ() {
        double nAccepted [][] = new double[nFramingAnnotators][nLabels];

        for (int k = 0; k < nFramingAnnotators; k++) {
            ArrayList<Integer> articles = framingAnnotatorArticles.get(k);

            for (int j = 0; j < nLabels; j++) {
                double current = q[k][j];
                // transform from (0.5,1) to R and back
                //double proposalReal = Math.log(-Math.log(current)) + rand.nextGaussian() * mhQSigma;
                //double proposal = Math.exp(-Math.exp(proposalReal));
                double proposal = current + rand.nextGaussian() * mhQSigma;

                double a;
                if (proposal > 0 && proposal < 1) {

                    /*
                    // using somewhat strong beta prior for now
                    double alpha = Transformations.betaMuGammaToAlpha(mu_q, gamma_q);
                    double beta = Transformations.betaMuGammaToBeta(mu_q, gamma_q);

                    if (alpha < 1 || beta < 1) {
                        System.out.println("Undesired beta parameters in Q");
                        alpha = Transformations.betaMuGammaToAlpha(mu_q, gamma_q);
                        beta = Transformations.betaMuGammaToBeta(mu_q, gamma_q);
                    }

                    BetaDistribution prior = new BetaDistribution(alpha, beta);
                    double pLogCurrent = prior.density(current);
                    double pLogProposal = prior.density(proposal);
                    */

                    // try using a U[0,1] prior and hope initialization guides us to identifiability
                    double pLogCurrent = 0.0;
                    double pLogProposal = 0.0;

                    for (int article : articles) {
                        int frames[] = articleFrames.get(article);
                        HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(article);
                        int labels[] = articleAnnotations.get(k);
                        double pPosCurrent = frames[j] * current + (1 - frames[j]) * (1 - r[k][j]);
                        double pPosProposal = frames[j] * proposal + (1 - frames[j]) * (1 - r[k][j]);
                        pLogCurrent += labels[j] * Math.log(pPosCurrent) + (1 - labels[j]) * Math.log(1 - pPosCurrent);
                        pLogProposal += labels[j] * Math.log(pPosProposal) + (1 - labels[j]) * Math.log(1 - pPosProposal);
                    }
                    a = Math.exp(pLogProposal - pLogCurrent);
                }
                else {
                    a = -1;
                }

                if (Double.isNaN(a)) {
                    System.out.println("NaN in Q:" + current + " to " + proposal);
                }
                if (Double.isInfinite(a)) {
                    System.out.println("Inf in Q:" + current + " to " + proposal);
                }

                double u = rand.nextDouble();
                if (u < a) {
                    q[k][j] = proposal;
                    nAccepted[k][j] += 1;
                }
            }
        }
        return nAccepted;
    }


    double [][] sampleR() {
        double nAccepted [][] = new double[nFramingAnnotators][nLabels];

        for (int k = 0; k < nFramingAnnotators; k++) {
            ArrayList<Integer> articles = framingAnnotatorArticles.get(k);

            for (int j = 0; j < nLabels; j++) {
                double current = r[k][j];
                // transform from (0.5,1) to R and back
                //double proposalReal = Math.log(-Math.log((current-0.5)*2)) + rand.nextGaussian() * mhQSigma;
                //double proposal = Math.exp(-Math.exp(proposalReal)) / 2 + 0.5;
                double proposal = current + rand.nextGaussian() * mhRSigma;

                double a;
                if (proposal > 0 && proposal < 1) {

                    /*
                    // using somewhat strong beta prior for now
                    //BetaDistribution prior = new BetaDistribution(q_mu * q_gamma / (1 - q_mu), q_gamma);
                    double alpha = Transformations.betaMuGammaToAlpha(mu_q, gamma_q);
                    double beta = Transformations.betaMuGammaToBeta(mu_q, gamma_q);

                    if (alpha < 1 || beta < 1) {
                        System.out.println("Undesired beta parameters in R");
                        alpha = Transformations.betaMuGammaToAlpha(mu_q, gamma_q);
                        beta = Transformations.betaMuGammaToBeta(mu_q, gamma_q);
                    }

                    BetaDistribution prior = new BetaDistribution(alpha, beta);

                    double pLogCurrent = prior.density(current);
                    double pLogProposal = prior.density(proposal);
                    */

                    double pLogCurrent = 0.0;
                    double pLogProposal = 0.0;

                    for (int article : articles) {
                        int frames[] = articleFrames.get(article);
                        HashMap<Integer, int[]> articleAnnotations = framingAnnotations.get(article);
                        int labels[] = articleAnnotations.get(k);
                        double pPosCurrent = frames[j] * q[k][j] + (1 - frames[j]) * (1 - current);
                        double pPosProposal = frames[j] * q[k][j] + (1 - frames[j]) * (1 - proposal);
                        pLogCurrent += labels[j] * Math.log(pPosCurrent) + (1 - labels[j]) * Math.log(1 - pPosCurrent);
                        pLogProposal += labels[j] * Math.log(pPosProposal) + (1 - labels[j]) * Math.log(1 - pPosProposal);
                    }
                    a = Math.exp(pLogProposal - pLogCurrent);

                    if (Double.isNaN(a)) {
                        System.out.println("NaN in R:" + current + " to " + proposal);
                    }
                    if (Double.isInfinite(a)) {
                        System.out.println("Inf in R:" + current + " to " + proposal);
                    }
                }
                else {
                    a = -1;
                }

                double u = rand.nextDouble();
                if (u < a) {
                    r[k][j] = proposal;
                    nAccepted[k][j] += 1;
                }
            }
        }
        return nAccepted;
    }


    double [][] sampleS() {
        double nAccepted [][] = new double[nToneAnnotators][nTones];

        for (int k = 0; k < nToneAnnotators; k++) {
            ArrayList<Integer> articles = toneAnnotatorArticles.get(k);

            for (int l = 0; l < nTones; l++) {
                double currentSimplex [] = sSimplex[k][l];
                double currentReals [] = sReal[k][l];

                MultivariateNormalDistribution multNormDist = new MultivariateNormalDistribution(currentReals, mhSCovariance);
                double proposalReal [] = multNormDist.sample();
                double proposalSimplex [] = Transformations.realsToSimplex(proposalReal, nTones);

                double a;

                if (proposalSimplex[l] > 0) {
                    // using pretty strong Dirichlet prior for now
                    double alpha[] = {1.0, 1.0, 1.0};
                    alpha[l] = 3;
                    DirichletDist prior = new DirichletDist(alpha);

                    double pLogCurrent = Math.log(prior.density(currentSimplex));
                    double pLogProposal = Math.log(prior.density(proposalSimplex));


                    //double pLogCurrent = 0.0;
                    //double pLogProposal = 0.0;

                    for (int article : articles) {
                        int tone = articleTone[article];
                        if (tone == l) {
                            HashMap<Integer, Integer> articleAnnotations = toneAnnotations.get(article);
                            int annotatorTone = articleAnnotations.get(k);
                            pLogCurrent += Math.log(currentSimplex[annotatorTone]);
                            pLogProposal += Math.log(proposalSimplex[annotatorTone]);
                        }
                    }
                    a = Math.exp(pLogProposal - pLogCurrent);

                    if (Double.isNaN(a)) {
                        System.out.println("NaN in S");
                    }
                    if (Double.isInfinite(a)) {
                        System.out.println("Inf in S");
                    }
                } else {
                    a = -1;
                }

                double u = rand.nextDouble();
                if (u < a) {
                    System.arraycopy(proposalReal, 0, sReal[k][l], 0, nTones);
                    System.arraycopy(proposalSimplex, 0, sSimplex[k][l], 0, nTones);
                    nAccepted[k][l] += 1;
                }
            }
        }
        return nAccepted;
    }



    private double[] recomputeGlobalMean() {
        double mean[] = new double[nLabels];
        // loop through all articles
        for (int i = 0; i < nArticlesWithFraming; i++) {
            // get the current article dist
            int articleLabels[] = articleFrames.get(i);
            for (int k = 0; k < nLabels; k++) {
                mean[k] += (double) articleLabels[k];
            }
        }
        for (int k = 0; k < nLabels; k++) {
            mean[k] = mean[k] / (double) nArticlesWithFraming;
        }
        return mean;
    }


    /*
    private double sampleTimeFramesGuassian() {
        double nAccepted = 0;

        // loop through all time points
        for (int t = 0; t < nTimes; t++) {

            // get the distribution over frames at the previous time point
            double previous[];
            if (t > 0) {
                previous = timeFrames.get(t-1);
            } else {
                previous = new double[nLabels];
                for (int k = 0; k < nLabels; k++) {
                    previous[k] = 1.0 / nLabels;
                }
            }

            // use this to compute a distribution over the current distribution over frames
            double previousDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                previousDist[k] = previous[k] * nLabels * alpha + alpha0;
            }

            // get the current distribution over frames
            double current[] = timeFrames.get(t);
            // create a variable for a proposal
            double proposal[] = new double[nLabels];

            // get the distribution over frames in the next time point
            double next[] = new double[nLabels];
            if (t < nTimes-1) {
                next = timeFrames.get(t + 1);
            }


            // generate a proposal (symmetrically)
            double temp[] = new double[nLabels];
            double total = 0;
            for (int k = 0; k < nLabels; k++) {
                temp[k] = Math.exp(Math.log(current[k]) + rand.nextGaussian() * mhTimeFrameSigma);
                total += temp[k];
            }
            for (int k = 0; k < nLabels; k++) {
                proposal[k] = temp[k] / total;
            }


            / *
            // compute a distribution for generating a proposal
            double mhProposalDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                mhProposalDist[k] = mhDirichletScale * current[k] + mhDirichletBias;
            }

            // generate a point from this distribution
            DirichletGen.nextPoint(randomStream, mhProposalDist, proposal);

            // evaluate the probability of generating this new point from the proposal
            DirichletDist dirichletDist = new DirichletDist(mhProposalDist);
            double mhpProposal = dirichletDist.density(proposal);

            // compute the probability of the generating the current point from the proposal
            double reverseDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                reverseDist[k] = mhDirichletScale * proposal[k] + mhDirichletBias;
            }

            DirichletDist dirichletDistReverse = new DirichletDist(reverseDist);
            double mhpReverse = dirichletDist.density(current);
            * /

            // compute a distribution over a new distribution over frames for the current distribution
            double currentDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                currentDist[k] = current[k] * nLabels * alpha + alpha0;
            }

            // do the same for the proposal
            double proposalDist[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                proposalDist[k] = proposal[k] * nLabels * alpha + alpha0;
            }

            // compute the probability of the current distribution over frames conditioned on the previous
            DirichletDist dirichletDistPrevious = new DirichletDist(previousDist);
            double pCurrentGivenPrev = dirichletDistPrevious.density(current);
            double pProposalGivenPrev = dirichletDistPrevious.density(proposal);

            // do the same for the next time point conditioned on the current and proposal
            DirichletDist dirichletDistCurrent = new DirichletDist(currentDist);
            double pNextGivenCurrent = dirichletDistCurrent.density(next);

            DirichletDist dirichletDistProposal = new DirichletDist(proposalDist);
            double pNextGivenProposal = dirichletDistProposal.density(next);

            double pLogCurrent = Math.log(pCurrentGivenPrev);
            if (t < t-1) {
                pLogCurrent += Math.log(pNextGivenCurrent);
            }
            double pLogProposal = Math.log(pProposalGivenPrev);
            if (t < t-1) {
                pLogProposal += Math.log(pNextGivenProposal);
            }

            double beta = betas.get(t);
            // compute distributions over distributions for articles
            double currentDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                currentDistArticle[k] = current[k] * nLabels * beta + beta0;
            }

            DirichletDist dirichletDistArticleCurrent = new DirichletDist(currentDistArticle);

            double proposalDistArticle[] = new double[nLabels];
            for (int k = 0; k < nLabels; k++) {
                proposalDistArticle[k] = proposal[k] * nLabels * beta + beta0;
            }

            DirichletDist dirichletDistArticleProposal = new DirichletDist(proposalDistArticle);

            // compute the probability of the article disrtibutions for both current and proposal
            ArrayList<Integer> articles = timeArticles.get(t);
            for (int i : articles) {
                double articleDist[] = articleFrames.get(i);
                double pArticleCurrent = dirichletDistArticleCurrent.density(articleDist);
                pLogCurrent += Math.log(pArticleCurrent);
                double pArticleProposal= dirichletDistArticleProposal.density(articleDist);
                pLogProposal += Math.log(pArticleProposal);
            }

            double a = Math.exp(pLogProposal - pLogCurrent);

            double u = rand.nextDouble();

            if (u < a) {
                timeFrames.set(t, proposal);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }
    */


}
