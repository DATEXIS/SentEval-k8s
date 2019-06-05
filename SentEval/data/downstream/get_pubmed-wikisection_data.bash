#!/usr/bin/env bash

data_path=.

### PubMedSection
mkdir $data_path/PubMedSection
curl -Lo $data_path/PubMedSection/data.tar.xz https://cloud.beuth-hochschule.de/index.php/s/46ejzpyRi2J9Ao8/download
tar -xf $data_path/PubMedSection/data.tar.xz -C $data_path/PubMedSection
rm $data_path/PubMedSection/data.tar.xz

### WikiSection
mkdir $data_path/WikiSection
curl -Lo $data_path/WikiSection/data.tar.xz https://cloud.beuth-hochschule.de/index.php/s/cj6xegeADpsmaTR/download
tar -xf $data_path/WikiSection/data.tar.xz -C $data_path/WikiSection
rm $data_path/WikiSection/data.tar.xz