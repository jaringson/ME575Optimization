import subprocess
import sys
sys.path.append('./supersonic-files')

import os
pwd = os.path.abspath(os.getcwd())
# print(os.getcwd())
# print(pwd)

# pwd = "/home/jaron/Documents/Classes/optimization_ME575/hmwk6"

# me575hw6 = subprocess.Popen("java -jar me575hw6.jar")
me575hw6 = subprocess.Popen("java -jar " + pwd + "/supersonic-files/me575hw6.jar")
# colt = subprocess.Popen("java -jar " + pwd + "/supersonic-files/colt.jar")
