import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Array;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import umontreal.ssj.rng.*;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.BetaDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.analysis.function.Sigmoid;


public class Sampler {

    private int nArticles;
    private int nAnnotators;
    private int nLabels;
    private int nTimes;
    private int nNewspapers;

    private static int firstYear = 1980;
    private static int nMonthsPerYear = 12;
    private static int nQuartersPerYear = 4;
    private static int nMonthsPerQuarter = 3;

    private HashMap<String, Integer> newspaperIndices;
    private HashMap<String, Integer> articleNameTime;
    private HashMap<String, Integer> articleNameNewspaper;
    private HashMap<Integer, Integer> articleTime;
    private HashMap<Integer, Integer> articleNewspaper;
    private HashMap<Integer, ArrayList<Integer>> timeArticles;
    //private HashMap<Integer, ArrayList<Integer>> newspaperArticles;
    private HashMap<String, Integer> annotatorIndices;
    private HashMap<Integer, ArrayList<Integer>> annotatorArticles;

    private ArrayList<String> articleNames;
    private ArrayList<HashMap<Integer, int[]>> annotations;

    private ArrayList<double[]> timeFramesReals;     // phi
    private ArrayList<double[]> timeFramesCube;
    private ArrayList<int[]> articleFrames ;  // theta
    private double[][] sens;
    private double[][] spec;
    private double[] framesMean;
    private double[] globalMean;

    private static double alphaTimes = 2.0;
    private static double timesRealSigma = 0.01;

    private static double mhTimeStepSigma = 0.1 ;
    private static double mhSensSigma = 0.4;
    private static double mhSpecSigma = 0.25;

    private static Random rand = new Random();
    private static RandomStream randomStream = new MRG32k3a();
    private static Sigmoid sigmoid = new Sigmoid();

    public Sampler(String inputFilename, String metadataFilename, String predictionsFilename) throws Exception {

        Path inputPath = Paths.get(inputFilename);
        JSONParser parser = new JSONParser();
        JSONObject data = (JSONObject) parser.parse(new FileReader(inputPath.toString()));

        Path metadataPath = Paths.get(metadataFilename);
        JSONObject metadata = (JSONObject) parser.parse(new FileReader(metadataPath.toString()));

        nLabels = 15;

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
            int quarter = (int) Math.floor((double) month / (double) nMonthsPerQuarter);
            int time = (year - firstYear) * nQuartersPerYear + quarter;
            if (time >= 0) {
                articleNameTime.put(articleName.toString(), time);
                articleNameNewspaper.put(articleName.toString(), newspaperIndices.get(source));
            }
            if (time >= nTimes) {
                nTimes = time + 1;
            }

        }

        System.out.println(nTimes + " time periods");

        timeArticles = new HashMap<>();
        for (int t = 0; t < nTimes; t++) {
            timeArticles.put(t, new ArrayList<>());
        }

        // index annotators
        Set<String> annotators = gatherAnnotators(data);
        nAnnotators = annotators.size();
        System.out.println(nAnnotators + " annotators");
        annotatorIndices = new HashMap<>();
        annotatorArticles = new HashMap<>();
        int j = 0;
        for (String annotator : annotators) {
            annotatorIndices.put(annotator, j);
            annotatorArticles.put(j, new ArrayList<>());
            j += 1;
        }
        System.out.println(annotatorIndices);

        // read in the annotations and build up all relevant information
        articleNames = new ArrayList<>();
        articleTime = new HashMap<>();
        articleNewspaper = new HashMap<>();
        annotations = new ArrayList<>();
        framesMean = new double[nLabels];
        int count = 0;
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
                    int time = articleNameTime.get(articleName.toString());
                    articleTime.put(i, time);
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
                        // store the total number of annotations for each label
                        annotatorArticles.get(annotatorIndex).add(i);
                        for (int k = 0; k < nLabels; k++) {
                            framesMean[k] += (double) annotationArray[k];
                        }
                        count += 1;
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                    timeArticles.get(time).add(i);
                }
            }
        }

        // read in predictions and add to annotations
        Scanner scanner = new Scanner(new File(predictionsFilename));
        HashMap<String, int[]> predictions = new HashMap<>();
        scanner.useDelimiter(",");
        for (int i = 0; i < 17; i++) {
            String next = scanner.next();
        }
        String rowArticleName = "";
        int[] pArray = new int[nLabels];
        int iNext = 0;
        while (scanner.hasNext()) {
            String next = scanner.next();
            int k = iNext % 17;
            if (k == 1) {
                rowArticleName = next;
            }
            else if (k > 1) {
                pArray[k-2] = Integer.parseInt(next);
            }
            if (k == 16) {
                predictions.put(rowArticleName, pArray);
                pArray = new int[nLabels];
            }
            iNext += 1;
        }

        //Do not forget to close the scanner
        scanner.close();

        /*
        int annotatorIndex = nAnnotators;
        annotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : predictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                JSONObject article = (JSONObject) data.get(articleName);
                // only use if we have no annotations for this article
                if (!articleNames.contains(articleName)) {
                    // get a new article id (i)
                    int i = articleNames.size();
                    // store the article name for future reference
                    articleNames.add(articleName);
                    // get the timestep for this article (by name)
                    int time = articleNameTime.get(articleName);
                    articleTime.put(i, time);
                    // get the newspaper for this article (by name)
                    articleNewspaper.put(i, articleNameNewspaper.get(articleName));
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = new HashMap<>();
                    // treat predictions as coming from a separate anntoator
                    // loop through this annotator's annotations
                    articleAnnotations.put(annotatorIndex, predictions.get(articleName));
                    // store the total number of annotations for each label
                    annotatorArticles.get(annotatorIndex).add(i);
                    for (int k = 0; k < nLabels; k++) {
                        framesMean[k] += (double) predictions.get(articleName)[k];
                    }
                    // store the annotations for this article
                    annotations.add(articleAnnotations);
                    timeArticles.get(time).add(i);
                    count += 1;
                }
            }
        }
        nAnnotators += 1;
        */

        // TEST: Only use articles that already have annotations!
        int annotatorIndex = nAnnotators;
        annotatorArticles.put(annotatorIndex, new ArrayList<>());
        for (String articleName : predictions.keySet()) {
            // check if this article is in the list of valid articles
            if (articleNameTime.containsKey(articleName)) {
                JSONObject article = (JSONObject) data.get(articleName);
                if (articleNames.contains(articleName)) {
                    // get the article ID
                    int i = articleNames.indexOf(articleName);
                    // create a hashmap to store the annotations for this article
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(i);
                    // treat predictions as coming from a separate annotator
                    articleAnnotations.put(annotatorIndex, predictions.get(articleName));
                    annotatorArticles.get(annotatorIndex).add(i);
                    // store the annotations for this article
                    annotations.set(i, articleAnnotations);
                }
            }
        }
        nAnnotators += 1;

        // don't bother using predictions for this...
        for (int k = 0; k < nLabels; k++) {
            framesMean[k] = framesMean[k] / (double) count;
        }

        nArticles = annotations.size();

        System.out.println("Mean of anntoations:");
        for (int k = 0; k < nLabels; k++) {
            System.out.print(framesMean[k] + " ");
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
        // initialize article frames based on mean of annotations
        articleFrames = new ArrayList<>();
        for (int i = 0; i < nArticles; i++) {
            int initialLabels[] = new int[nLabels];
            double annotationCounts[] = new double[nLabels];
            // add the contributions of each annotator
            HashMap<Integer, int[]> articleAnnotations = annotations.get(i);
            double nAnnotators = (double) articleAnnotations.size();
            for (int annotator : articleAnnotations.keySet()) {
                int annotatorAnnotations[] = articleAnnotations.get(annotator);
                for (int k = 0; k < nLabels; k++) {
                    annotationCounts[k] += (double) annotatorAnnotations[k] / nAnnotators;
                }
            }
            for (int k = 0; k < nLabels; k++) {
                double u = rand.nextDouble();
                if (u < annotationCounts[k]) {
                    initialLabels[k] = 1;
                }
            }
            articleFrames.add(initialLabels);
        }

        // initialize timeFrames as the mean of the corresponding articles
        timeFramesReals = new ArrayList<>();
        timeFramesCube = new ArrayList<>();
        for (int t = 0; t < nTimes; t++) {
            double timeMean[] = new double[nLabels];
            double timeMeanReals[] = new double[nLabels];
            // initialize everything with the global mean
            ArrayList<Integer> articles = timeArticles.get(t);
            double nTimeArticles = (double) articles.size();
            for (int k = 0; k < nLabels; k++) {
                timeMean[k] += framesMean[k] / (nTimeArticles + 1);
            }
            for (int i : articles) {
                int frames[] = articleFrames.get(i);
                for (int k = 0; k < nLabels; k++) {
                    timeMean[k] += (double) frames[k] / (nTimeArticles + 1);
                }
            }

            for (int k = 0; k < nLabels; k++) {
                //timeMean[k] = 1.0 / (double) nLabels;
                timeMeanReals[k] = Math.log(-Math.log(timeMean[k]));
            }
            timeFramesCube.add(timeMean);
            timeFramesReals.add(timeMeanReals);
        }

        System.out.println("");

        // initialize annotator parameters to reasonable values
        sens = new double[nAnnotators][nLabels];
        spec = new double[nAnnotators][nLabels];
        for (int j = 0; j < nAnnotators; j++) {
            for (int k = 0; k < nLabels; k++) {
                sens[j][k] = 0.7;
                spec[j][k] = 0.7;
            }
        }
    }

    void run(int nIter, int burnIn, int samplingPeriod, int printPeriod) throws  Exception {
        int nSamples = (int) Math.floor((nIter - burnIn) / (double) samplingPeriod);

        double timeFrameSamples [][][] = new double[nSamples][nTimes][nLabels];
        int articleFrameSamples [][][] = new int[nSamples][nArticles][nLabels];
        double sensSamples [][][] = new double[nSamples][nAnnotators][nLabels];
        double specSamples [][][] = new double[nSamples][nAnnotators][nLabels];

        double timeFrameRate = 0.0;
        double articleFramesRate = 0;
        double sensRate = 0;
        double specRate = 0;
        int s = 0;
        int i = 0;
        while (s < nSamples) {
            timeFrameRate += sampleTimeFrames();
            articleFramesRate += sampleArticleFrames();

            // Not so bad to have one per annotator, but still seems to be better without
            // Also, high rejection rates for these... try Guassians?
            sensRate += sampleSens();
            specRate += sampleSpec();

            globalMean = recomputeGlobalMean();

            // save samples
            if (i > burnIn && i % samplingPeriod == 0) {
                for (int t = 0; t < nTimes; t++) {
                    System.arraycopy(timeFramesCube.get(t), 0, timeFrameSamples[s][t], 0, nLabels);
                }
                for (int a = 0; a < nArticles; a++) {
                    System.arraycopy(articleFrames.get(a), 0, articleFrameSamples[s][a], 0, nLabels);
                }
                for (int j = 0; j < nAnnotators; j++) {
                    System.arraycopy(sens[j], 0, sensSamples[s][j], 0, nLabels);
                    System.arraycopy(spec[j], 0, specSamples[s][j], 0, nLabels);
                }
                s += 1;
            }
            i += 1;
            if (i % printPeriod == 0) {
                System.out.print(i + ": ");
                for (int k = 0; k < nLabels; k++) {
                    System.out.print(globalMean[k] + " ");
                }
                System.out.println("");
            }
        }

        System.out.println(timeFrameRate / nIter);
        System.out.println(articleFramesRate/ nIter);
        System.out.println(sensRate / nIter);
        System.out.println(specRate / nIter);

        // save results

        Path output_path;
        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "timeFramesSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int t = 0; t < nTimes; t++) {
                        file.write(timeFrameSamples[s][t][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "sensSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int j = 0; j < nAnnotators; j++) {
                        file.write(sensSamples[s][j][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "specSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int j = 0; j < nAnnotators; j++) {
                        file.write(specSamples[s][j][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }

        /*
        output_path = Paths.get("samples", "sensSamples.csv");
        try (FileWriter file = new FileWriter(output_path.toString())) {
            for (s = 0; s < nSamples; s++) {
                for (int j = 0; j < nAnnotators; j++) {
                    for (int k = 0; k < nLabels; k++) {
                        file.write(specSamples[s][j][k] + ",");
                    }
                }
                file.write("\n");
            }
        }



        for (int k = 0; k < nLabels; k++) {
            output_path = Paths.get("samples", "articleSamples" + k + ".csv");
            try (FileWriter file = new FileWriter(output_path.toString())) {
                for (s = 0; s < nSamples; s++) {
                    for (int a = 0; a < nArticles; a++) {
                        file.write(articleFrameSamples[s][a][k] + ",");
                    }
                    file.write("\n");
                }
            }
        }
        */

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
            double step = rand.nextGaussian() * mhTimeStepSigma;

            // apply the step to generate a proposal
            for (int k = 0; k < nLabels; k++) {
                proposalReals[k] = currentReals[k] + normalStep[k] * step;
            }

            // transform the proposal to the cube
            double proposalCube[] = realsToCube(proposalReals, nLabels);

            // get the distribution over frames at the previous time point
            double previousCube[];
            double previousReals[];
            if (t > 0) {
                previousCube = timeFramesCube.get(t-1);
                previousReals = timeFramesReals.get(t-1);
            } else {
                // if t == 0, use the global mean
                previousCube = new double[nLabels];
                System.arraycopy(framesMean, 0, previousCube, 0, nLabels);
                previousReals = cubeToReals(previousCube, nLabels);
            }

            // compute a distribution over the current time point given previous
            double priorCovariance[][] = new double[nLabels][nLabels];
            for (int k = 0; k < nLabels; k++) {
                priorCovariance[k][k] = timesRealSigma;
            }

            MultivariateNormalDistribution previousDist = new MultivariateNormalDistribution(previousReals, priorCovariance);

            double pLogCurrent = Math.log(previousDist.density(currentReals));
            double pLogProposal = Math.log(previousDist.density(proposalReals));

            // get the distribution over frames in the next time point
            if (t < nTimes-1) {
                double nextReals[] = timeFramesReals.get(t+1);

                // compute a distribution over a new distribution over frames for the current distribution
                MultivariateNormalDistribution currentDist = new MultivariateNormalDistribution(currentReals, priorCovariance);
                MultivariateNormalDistribution proposalDist = new MultivariateNormalDistribution(proposalReals, priorCovariance);

                pLogCurrent += Math.log(currentDist.density(nextReals));
                pLogProposal += Math.log(proposalDist.density(nextReals));
            }

            // compute the probability of the article distributions for both current and proposal
            ArrayList<Integer> articles = timeArticles.get(t);
            for (int i : articles) {
                int articleLabels[] = articleFrames.get(i);
                for (int k = 0; k < nLabels; k++) {
                    pLogCurrent += articleLabels[k] * Math.log(currentCube[k]) + (1-articleLabels[k]) * Math.log(1-currentCube[k]);
                    pLogProposal += articleLabels[k] * Math.log(proposalCube[k]) + (1-articleLabels[k]) * Math.log(1-proposalCube[k]);
                }
            }

            double a = Math.exp(pLogProposal - pLogCurrent);
            double u = rand.nextDouble();

            if (u < a) {
                timeFramesCube.set(t, proposalCube);
                timeFramesReals.set(t, proposalReals);
                nAccepted += 1;
            }

        }
        return nAccepted / nTimes;
    }


    private double sampleArticleFrames() {
        double nAccepted = 0;

        // loop through all articles
        for (int i = 0; i < nArticles; i++) {

            // get the current article dist
            int articleLabels[] = articleFrames.get(i);
            int proposedLabels[] = new int[nLabels];

            int time = articleTime.get(i);
            double timeFrames[] = timeFramesCube.get(time);

            for (int k = 0; k < nLabels; k++) {

                double pLogPosGivenTime = Math.log(timeFrames[k]);
                double pLogNegGivenTime = Math.log(1-timeFrames[k]);

                // compute the probability of the labels for both current and proposal
                HashMap<Integer, int[]> articleAnnotations = annotations.get(i);
                for (int annotator : articleAnnotations.keySet()) {
                    int labels[] = articleAnnotations.get(annotator);
                    pLogPosGivenTime += labels[k] * Math.log(sens[annotator][k]) + (1-labels[k]) * Math.log(1-spec[annotator][k]);
                    pLogNegGivenTime += labels[k] * Math.log(1-sens[annotator][k]) + (1-labels[k]) * Math.log(spec[annotator][k]);
                }

                double pPosUnnorm = Math.exp(pLogPosGivenTime);
                double pNegUnnorm = Math.exp(pLogNegGivenTime);
                double pPos = pPosUnnorm / (pPosUnnorm + pNegUnnorm);

                double u = rand.nextDouble();

                if (u < pPos) {
                    proposedLabels[k] = 1;
                }
            }
            articleFrames.set(i, proposedLabels);
        }
        return nAccepted / nArticles;
    }


    private double sampleSens() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            for (int k = 0; k < nLabels; k++) {

                ArrayList<Integer> articles = annotatorArticles.get(annotator);

                // get the current value
                double current = sens[annotator][k];
                double currentReal = Math.log(-Math.log(current));

                NormalDistribution normDist = new NormalDistribution(0, mhSensSigma);
                double proposalReal = currentReal + normDist.sample();
                double proposal = Math.exp(-Math.exp(proposalReal));

                BetaDistribution sensPrior = new BetaDistribution(2.0, 2.0/0.75 - 2.0);

                if (proposal >= 1.0) {
                    proposal = 1.0 - 1E-6;
                }

                double pLogCurrent = Math.log(sensPrior.density(current));
                double pLogProposal = Math.log(sensPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles
                for (int article : articles) {
                    int frames[] = articleFrames.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pPosCurrent = frames[k] * current + (1-frames[k]) * (1-spec[annotator][k]);
                    double pPosProposal = frames[k] * proposal + (1-frames[k]) * (1-spec[annotator][k]);
                    pLogCurrent += labels[k] * Math.log(pPosCurrent) + (1 - labels[k]) * Math.log(1 - pPosCurrent);
                    pLogProposal += labels[k] * Math.log(pPosProposal) + (1 - labels[k]) * Math.log(1 - pPosProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);
                double u = rand.nextDouble();

                if (u < a) {
                    sens[annotator][k] = proposal;
                    nAccepted += 1;
                }
            }
        }
        return nAccepted / (nAnnotators * nLabels);
    }

    private double sampleSpec() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            for (int k = 0; k < nLabels; k++) {

                ArrayList<Integer> articles = annotatorArticles.get(annotator);

                // get the current value
                double current = spec[annotator][k];

                double currentReal = Math.log(-Math.log(current));

                NormalDistribution normDist = new NormalDistribution(0, mhSpecSigma);
                double proposalReal = currentReal + normDist.sample();
                double proposal = Math.exp(-Math.exp(proposalReal));

                BetaDistribution sensPrior = new BetaDistribution(2.0, 2.0/0.75 - 2.0);

                if (proposal >= 1.0) {
                    proposal = 1.0 - 1E-6;
                }

                double pLogCurrent = Math.log(sensPrior.density(current));
                double pLogProposal = Math.log(sensPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles
                for (int article : articles) {
                    int frames[] = articleFrames.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pPosCurrent = frames[k] * sens[annotator][k] + (1-frames[k]) * (1-current);
                    double pPosProposal = frames[k] * sens[annotator][k] + (1-frames[k]) * (1-proposal);
                    pLogCurrent += labels[k] * Math.log(pPosCurrent) + (1 - labels[k]) * Math.log(1 - pPosCurrent);
                    pLogProposal += labels[k] * Math.log(pPosProposal) + (1 - labels[k]) * Math.log(1 - pPosProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);
                double u = rand.nextDouble();

                if (u < a) {
                    spec[annotator][k] = proposal;
                    nAccepted += 1;
                }
            }
        }
        return nAccepted / (nAnnotators * nLabels);
    }


    /*
    private double sampleZeal() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            for (int k = 0; k < nLabels; k++) {

                ArrayList<Integer> articles = annotatorArticles.get(annotator);

                // get the current value
                double current = zeal[annotator][k];

                //GammaDistribution proposalDist = new GammaDistribution(current+1, 1.0);
                //double proposal = proposalDist.run();
                //double mhpProposal = proposalDist.density(proposal);

                //GammaDistribution reverseDist = new GammaDistribution(proposal+1, 1.0);
                //double mhpReverse = reverseDist.density(current);

                double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhZealSigma);

                GammaDistribution zealPrior = new GammaDistribution(zealShape, zealScale);
                double pLogCurrent = Math.log(zealPrior.density(current));
                double pLogProposal = Math.log(zealPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles

                for (int article : articles) {
                    double articleDist[] = articleFramesSimplex.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * current - bias[annotator][k]);
                    double pLabelProposal = sigmoid.value(articleDist[k] * proposal - bias[annotator][k]);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);
                double u = rand.nextDouble();

                if (u < a) {
                    zeal[annotator][k] = proposal;
                    nAccepted += 1;
                }
            }
        }
        return nAccepted / (nAnnotators * nLabels);
    }


    private double sampleBias() {
        double nAccepted = 0;

        // loop through all annotators and labels
        for (int annotator = 0; annotator < nAnnotators; annotator++) {

            for (int k = 0; k < nLabels; k++) {

                ArrayList<Integer> articles = annotatorArticles.get(annotator);

                // get the current value
                double current = bias[annotator][k];
                // create a variable for a proposal
                //GammaDistribution proposalDist = new GammaDistribution(current, 1.0);
                //double proposal = proposalDist.run();
                //double mhpProposal = proposalDist.density(proposal);

                //GammaDistribution reverseDist = new GammaDistribution(proposal, 1.0);
                //double mhpReverse = reverseDist.density(current);
                double proposal = Math.exp(Math.log(current) + rand.nextGaussian() * mhBiasSigma);

                GammaDistribution biasPrior = new GammaDistribution(biasShape, biasScale);
                double pLogCurrent = Math.log(biasPrior.density(current));
                double pLogProposal = Math.log(biasPrior.density(proposal));

                // compute the probability of the annotations for all relevant articles
                for (int article : articles) {
                    double articleDist[] = articleFramesSimplex.get(article);
                    HashMap<Integer, int[]> articleAnnotations = annotations.get(article);
                    int labels[] = articleAnnotations.get(annotator);
                    double pLabelCurrent = sigmoid.value(articleDist[k] * zeal[annotator][k] - current);
                    double pLabelProposal = sigmoid.value(articleDist[k] * zeal[annotator][k] - proposal);
                    pLogCurrent += labels[k] * Math.log(pLabelCurrent) + (1 - labels[k]) * Math.log(1 - pLabelCurrent);
                    pLogProposal += labels[k] * Math.log(pLabelProposal) + (1 - labels[k]) * Math.log(1 - pLabelProposal);
                }

                double a = Math.exp(pLogProposal - pLogCurrent);
                double u = rand.nextDouble();

                if (u < a) {
                    bias[annotator][k] = proposal;
                    nAccepted += 1;
                }
            }
        }
        return nAccepted / (nAnnotators);
    }
    */


    private double[] recomputeGlobalMean() {
        double mean[] = new double[nLabels];
        // loop through all articles
        for (int i = 0; i < nArticles; i++) {
            // get the current article dist
            int articleLabels[] = articleFrames.get(i);
            for (int k = 0; k < nLabels; k++) {
                mean[k] += (double) articleLabels[k];
            }
        }
        for (int k = 0; k < nLabels; k++) {
            mean[k] = mean[k] / (double) nArticles;
        }
        return mean;
    }

    private double[] cubeToReals(double p[], int size) {
        double reals[] = new double[size];
        for (int k = 0; k < size; k++) {
            reals[k] = Math.log(-Math.log(p[k]));
        }
        return reals;
    }

    private double[] realsToCube(double r[], int size) {
        double cube[] = new double[size];
        for (int k = 0; k < size; k++) {
            cube[k] = Math.exp(-Math.exp(r[k]));
        }
        return cube;
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
