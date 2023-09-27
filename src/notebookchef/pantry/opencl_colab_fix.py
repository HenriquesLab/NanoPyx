#@title Fix OpenCL if needed in Google Colab
import sys

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    !sudo apt-get update -qq
    !sudo apt-get purge -qq *nvidia* -y
    !sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq nvidia-driver-530 -y
    exit(0)