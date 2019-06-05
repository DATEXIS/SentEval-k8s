# SentEval-k8s

A Kubernetes job to evaluate REST encoders with [SentEval](https://github.com/facebookresearch/SentEval/pulls).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To build, deploy and run the image you´ll need to have:

- docker with nvidia-docker
- curl
- sed
- awk
- zip (unzip)
- python3
- python3-pip

installed.

### Installing

#### Docker Image

To build the docker container, simply run the supplied build script `build.sh`.

#### Local

For local testing, install the following dependencies manually:

```bash
pip3 install scikit-learn sklearn torch requests
pip3 install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Running the tests

To execute the test for the core functionalities run

```bash
python3 -m unittest tests/test_evaluaterestencoder.py
```

## Deployment

This image is meant to be run as a Kubernetes job.
The configuration (encoder url, parameter, ...) has to be done by environment variables.
See `run_local.sh` or `run_k8sjob.sh` for details.
Upon completion, the results are saved as JSON in the corresponding *results* folder mount.

#### Deploy Image on DATEXIS Registry

An example of how to deploy the image onto the datexis registry can be found in `deploy.sh`. Don´t forget to login into the registry before deployment and change the namespace according to your configuration.

#### Run Job on K8s

To start a job on the K8s cluster, change the namespaces within `k8s-senteval-jobs.yaml` and run 

```bash
kubectl create -f k8s-senteval-job.yaml
```
or `run_k8sjob.sh`

#### Run Local Docker Image

See `run_local.sh` for an example.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/devfoo-one/SentEval-k8s/tags). 

## Authors

* **[Tom Oberhauser](https://github.com/devfoo-one)** - *Initial work*
* **[Paul Grundmann](https://github.com/Sunkua)**     - *Integration of PubMedSection and WikiSection tasks*

## License

TODO




