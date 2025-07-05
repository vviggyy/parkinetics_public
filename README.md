NOTE: This repository is a display (public) copy of a private team repo. Sensitive data has been removed. Contributors to this repo:
* Viggy Vanchinathan (Team Lead)
* Matthew Farah
* Arihant Singh
* Nana Osei-Owusu
Other team members:
* Chloe Zhang
* Ramya Palani
* Roma Desai 

# parkinetics

## release notes (gyro), vv 3/2/25
* added getters for x,y,z components of acc and gyro
* expanded viz.py
    * "component dashboard" --> x, y, z, mag for both accel and gyro
    * KL div is back
    * TSNE
    
## release notes (mappings), vv 2/15/25
* added functionality to automatically join a spreadsheet/csv of clinical metrics (e.g. updrs subscores) to an existing csv of signal features
* command-line tool called very similarly to pipeline.py
    * after getting an output by pipeline.py, call map.py with arguments as outlined below
* flag `-a` that allows to you to toggle merging both raw clinical subscores and calculated metrics (e.g. hand score, total updrs score) or JUST calculated metrics.
* TODOs
    * make sure the column names aren't hardcoded somehow. Include a CONSTANTS file somewhere?
    * assertions

## parkinetics pipeline mk II 
(release notes), vv 12/11/24

* major overhaul to design of pipeline
    * --> transition to object-oriented makes things much easier to
    * Data is no longer stored in tuples, it's stored in a dedicated `Zignal` object, which contains the movement data signal itself, along with it's associated object (e.g. participant, date, fs, trial type, etc.)
        * the `Zignal` class also contains methods to process an individual signal's data, such as `mag()`, `enmo()`, and `rm_out()` (remove outliers)
        * it's called `Zignal` because `Signal` was already taken by some other python standard library lol
    * Every time we run the pipeline, we generate a `Zignal` for each file we pass in. All `Zignals`
    in the pipeline run get stored in a new `Dataset` object
        * the `Dataset` object contains a list of `Zignals` and other methods to manage the collection together, such as `crop_sig()`.
        * it also contains a dictionary that stores each `Zignal` feature vector thats calculated manually. This gets made into the output metrics table.
    * Functions have largely been detangled, separated, and organized into respective locations
        * `viz.py` contains everything needed for visualization
        * `features.py` contains everything needed to calculate features. It's called by `viz.py`
        * `processing.py` contains all file processing functions, including `load_files()`, `fold_dfs()`, and a new function `process_file.py`
        * `pipeline.py` is where the actual magic happens. Call this function instead of `metrics.py` 
            * file flags remain the same. However, you need to now explicitly call `-m [any string]` to generate the metrics files.

* some other new features
    * Every single function is documented and commented. 
    * no longer do we have to maintain `.txt` files that have all the file paths we need. You can now pass in file FOLDERS into the `-o` flag, and the pipeline will analyze every .csv file within that folder, all the way to the bottom of the filetree. 
        * e.g. if i pass in `-o data/brush` I'll quickly generate metrics for ALL brushing trials, ever, no need to manually copy all the file paths.
            * this is particularly helpful for runs when we need to generate metrics quickly, so we don't have to maintain `all_data.txt`
        * Passing in `.txt` files still works though. This way, you can analyze/visualize trials in different data folders
    * The dependencies should be fixed. It really should just work out of the box if you set up the environment correctly, as I've written below

* some limitations and further TODOs:
    * need to add back tsfel and pca (only manual features are present for now)
    * pipeline doesn't currently analyze .txt files, but all .csv files are generated now anyway
    * i've only included (unnormalized) kde plots and a histogram in the viz file, for now
        * can be expanded later easily.

## How to set up

1. Ensure dependencies are correct
    * If you have conda, miniconda, or mamba installed:
        * `$ conda env create -f parkinetics.yml`
        * `$ conda activate parkinetics`
    * If you don't (or don't know what that is):
        * `$ pip install -r requirements.txt`

2. Run the pipeline 
    * for pipeline v1
        * `$ python metrics.py -i data.txt -p y -f 10 -o output.csv`
            * -i (input; add path to text file with the paths of files to be analyzed)
            * -p (plot; any string input toggles whether you generate it or not)
            * -f (fold; input length of folds in seconds)
            * -o (output; include path .csv file to upload data to. find the file within out/)
    
    * for pipeline v2
        * `$ python pipeline.py -i data/brush -m yes -p yes -f 7 -o test.csv`
            * -i (input; can be text file or directory)
            * -m (metrics: any string input toggles whether you generate a file or not)
            * -p (plot: any string input toggles whether you generate visualizations or not)
            * -f (fold: input length of folds in seconds)
            * -o (output; include name of .csv file you want to upload data to. find the file within out/ directory.)
            * -v (verbose; any string input will toggle a table outputted to the log with descriptive statistics on mean and std of features; use for sanity check)

    * join clinical metrics with signal features (NEW!)
        * (after getting an output from `pipeline.py`) run:
        * `$ python map.py -i out/<METRICS.CSV> -c <CLINICAL_SCORE_SPREADSHEET.csv> -a yes -o out/mappings/<OUTPUT.csv>`
            * -i (input; needs to be a .csv feature file that's generated with pipeline.py)
            * -c (clinical; needs to be a spreadsheet with raw clinical scores, each row corresponds to a participant)
            * -a (all; any string input toggles whether you join both raw subscores + calculated metrics)
            * -o (out; include name of output .csv file. You'll find it in the out/mappings directory.)
            * can either add all the metrics and clinical metrics together (INCLUDING PROCESSED) or simply the metrics and processed 
            clinical metrics
    * run regression models using clinical metrics-features together; pass in this file and -all and you'll see regression done with
    various selected outputs (graphs displaying correlation will also be shown concurrently)

Additional Points:
* the `:R` drive should be accessible through the path `\\ritnas.nas.jh.edu\parkinetic_data`. I've included a `.txt` file that you can run that will access data from :R\ automatically
