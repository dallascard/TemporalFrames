import java.util.HashMap;

public class Main {

    public static void main(String[] args) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "input/documents.json");   // documents.json
        params.put("-m", "input/summary.json");                    // metadata summary.json
        params.put("-p", "input/uncorrected_with_tone2.csv");                                                     // n_personas
        params.put("-x", "input/quarter_smoothed_dummy.csv");            // mood: quarter_smoothed_dummy.csv
        params.put("-i", "5000");            // number of iterations
        params.put("-b", "4000");            // burn in
        params.put("-s", "10");            // sampling period
        params.put("-v", "100");            // display period

        String arg = null;
        for (String s: args) {
            if (arg == null)
                arg = s;
            else {
                params.put(arg, s);
                arg = null;
            }
        }


        System.out.println(params);

        String documents = params.get("-d");
        String metadata = params.get("-m");
        String predictions = params.get("-p");
        String mood = params.get("-x");
        int n_iter = Integer.parseInt(params.get("-i"));
        int burn_in = Integer.parseInt(params.get("-b"));
        int sampling_period = Integer.parseInt(params.get("-s"));
        int display_period = Integer.parseInt(params.get("-v"));

        //String documents = "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json";
        //String metadata = "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json";
        //String predictions = "/Users/dcard/Desktop/uncorrected.csv";
        //String predictionsWithTone = "/Users/dcard/Desktop/uncorrected_with_tone.csv";
        //String mood = "/Users/dcard/Dropbox/CMU/ARK/compuframes/Analysis/Amber/data/quarter_smoothed_dummy.csv";
        //Sampler sampler = new Sampler(documents, metadata, predictions);


        boolean normalizeStoriesAtTime = true;
        boolean normalizeMood = true;

        CombinedModel sampler = new CombinedModel(documents, metadata, predictions, mood, normalizeStoriesAtTime, normalizeMood);

        sampler.run(n_iter, burn_in, sampling_period, display_period);

    }
}
