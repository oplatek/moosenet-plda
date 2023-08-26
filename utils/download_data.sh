#!/usr/bin/env bash
set -euo pipefail
set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

DATA_DIR=$SCRIPT_DIR/../data
VCC2018_DIR=$DATA_DIR/vcc2018/

cd  $VCC2018_DIR
printf "Downloading vcc2018 data\n\n"
# See https://datashare.ed.ac.uk/handle/10283/3257
wget https://datashare.ed.ac.uk/download/DS_10283_3257.zip
# See https://datashare.ed.ac.uk/handle/10283/3061
wget https://datashare.ed.ac.uk/download/DS_10283_3061.zip
printf "\n\nDone-Downloading vcc2018 data\n\n"


printf "Unzipping vcc2018 data\n\n"
unzip -f DS_10283_3257.zip
unzip -f DS_10283_3061.zip
printf "\n\nDone-Unzipping vcc2018 data\n\n"

cd -



# Copied from MOSNet
# download evaluation restuls
wget -O vcc2018_listening_test_scores.zip "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3257/vcc2018_listening_test_scores.zip?sequence=1&isAllowed=y" 
unzip vcc2018_listening_test_scores.zip
rm vcc2018_listening_test_scores.zip

# download submitted_systems_converted_speech
wget -O vcc2018_submitted_systems_converted_speech.tar.gz "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz?sequence=10&isAllowed=y" 
tar zxvf vcc2018_submitted_systems_converted_speech.tar.gz
rm vcc2018_submitted_systems_converted_speech.tar.gz
