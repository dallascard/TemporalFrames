import java.util.HashMap;

public class Main {

    public static void main(String[] args) throws Exception {

        HashMap<String, String> params = new HashMap<>();

        // set defaults
        params.put("-d", "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json");   // documents.json
        params.put("-m", "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json");                    // metadata summary.json
        params.put("-p", "/Users/dcard/Desktop/uncorrected_with_tone.csv");                                                     // n_personas
        params.put("-x", "/Users/dcard/Dropbox/CMU/ARK/compuframes/Analysis/Amber/data/quarter_smoothed_dummy.csv");            // mood: quarter_smoothed_dummy.csv

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

        //String documents = "/Users/dcard/Documents/Mercurial/compuframes-coding/tools/annotations/immigration/documents.json";
        //String metadata = "/Users/dcard/Dropbox/CMU/ARK/compuframes/data/metadata/immigration/summary.json";
        //String predictions = "/Users/dcard/Desktop/uncorrected.csv";
        //String predictionsWithTone = "/Users/dcard/Desktop/uncorrected_with_tone.csv";
        //String mood = "/Users/dcard/Dropbox/CMU/ARK/compuframes/Analysis/Amber/data/quarter_smoothed_dummy.csv";
        //Sampler sampler = new Sampler(documents, metadata, predictions);


        CombinedModel sampler = new CombinedModel(documents, metadata, predictions, mood);

        sampler.run(4000, 3000, 10, 100);

    }
}
