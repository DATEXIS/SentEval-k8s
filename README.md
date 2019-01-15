# SentEval-k8s

## Usage

1. Build image by running `build.sh`
2. Specify encoder-url in `run.sh`
3. Specify encoder-type (*TOKEN* or *SENTENCE*) in `run.sh`
3. For token encoders: specify aggregation (*AVG* or *ARORA*) in `run.sh`
4. Run `run.sh`
5. Upon completion, the results are saved as JSON in the corresponding *results* folder mount.