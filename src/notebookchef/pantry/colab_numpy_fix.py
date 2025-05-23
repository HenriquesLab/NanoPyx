#@title Fix numpy version in Google Colab, can be skipped if running in local environment. Session will now restart automatically. You can then proceed to the next cell.
import sys
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    !pip install -q numpy==1.26.4
    !pip install -q mako==1.3.0

    print("Session will now restart automatically. You can then proceed to the next cell.")
    import os
    os.kill(os.getpid(), 9)