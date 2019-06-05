#!/usr/bin/env bash

data_path=.

### PubMedSection
mkdir $data_path/PubMedSection
curl -Lo $data_path/PubMedSection/data.tar.xz https://cloud.beuth-hochschule.de/index.php/s/YGWojWKBKg66fnZ
tar -xf $data_path/PubMedSection/data.tar.xz -C $data_path/PubMedSection
rm $data_path/PubMedSection/data.tar.xz

### WikiSection
mkdir $data_path/WikiSection
curl -Lo $data_path/WikiSection/data.tar.xz https://cloud.beuth-hochschule.de/index.php/s/cQJgjcnx4KefXyp
tar -xf $data_path/WikiSection/data.tar.xz -C $data_path/WikiSection
rm $data_path/WikiSection/data.tar.xz